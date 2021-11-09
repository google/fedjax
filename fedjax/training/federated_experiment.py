# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Federated experiment manager."""

import abc
import os.path
import time
from typing import Any, Mapping, NamedTuple, Optional, Sequence, Tuple

from absl import logging
from fedjax.core import client_datasets
from fedjax.core import client_samplers
from fedjax.core import federated_algorithm
from fedjax.core import federated_data
from fedjax.core import models
from fedjax.core import util
from fedjax.training import checkpoint
from fedjax.training import logging as fedjax_logging
import jax.numpy as jnp

tf = util.import_tf()


def set_tf_cpu_only():
  """Restricts TensorFlow device visibility to only CPU.

  TensorFlow is only used for data loading, so we prevent it from allocating
  GPU/TPU memory.
  """
  tf.config.experimental.set_visible_devices([], 'GPU')
  tf.config.experimental.set_visible_devices([], 'TPU')


class EvaluationFn(metaclass=abc.ABCMeta):
  """Evaluation function that are only fed state at every call.

  Typically used for full evaluation or evaluation on sampled clients from a
  test set.
  """

  @abc.abstractmethod
  def __call__(self, state: federated_algorithm.ServerState,
               round_num: int) -> Mapping[str, jnp.ndarray]:
    """Runs final evaluation."""


class ModelSampleClientsEvaluationFn(EvaluationFn):
  """Evaluation on sampled clients using the centralized model.

  The state to be evaluated must contain a params field.
  """

  def __init__(self, client_sampler: client_samplers.ClientSampler,
               model: models.Model,
               batch_hparams: client_datasets.PaddedBatchHParams):
    self._client_sampler = client_sampler
    self._model = model
    self._batch_hparams = batch_hparams

  def __call__(self, state: federated_algorithm.ServerState,
               round_num: int) -> Mapping[str, jnp.ndarray]:
    params = state.params
    self._client_sampler.set_round_num(round_num)
    clients = self._client_sampler.sample()
    batches = client_datasets.padded_batch_client_datasets(
        [i for _, i, _ in clients], self._batch_hparams)
    return models.evaluate_model(self._model, params, batches)


class ModelFullEvaluationFn(EvaluationFn):
  """Evaluation on an entire federated dataset using the centralized model."""

  def __init__(self, fd: federated_data.FederatedData, model: models.Model,
               batch_hparams: client_datasets.PaddedBatchHParams):
    self._fd = fd
    self._model = model
    self._batch_hparams = batch_hparams

  def __call__(self, state: federated_algorithm.ServerState,
               round_num: int) -> Mapping[str, jnp.ndarray]:
    del round_num
    params = state.params
    batches = federated_data.padded_batch_federated_data(
        self._fd, self._batch_hparams)
    return models.evaluate_model(self._model, params, batches)


class TrainClientsEvaluationFn(metaclass=abc.ABCMeta):
  """Evaluation function that are fed training clients at every call.

  Typically used for evaluation on the training clients used in a step.
  """

  @abc.abstractmethod
  def __call__(
      self, state: federated_algorithm.ServerState, round_num: int,
      train_clients: Sequence[Tuple[federated_data.ClientId,
                                    client_datasets.ClientDataset, Any]]
  ) -> Mapping[str, jnp.ndarray]:
    """Runs evaluation."""


class ModelTrainClientsEvaluationFn(TrainClientsEvaluationFn):
  """Evaluation on training clients using the centralized model.

  The state to be evaluated must contain a params field.
  """

  def __init__(self, model: models.Model,
               batch_hparams: client_datasets.PaddedBatchHParams):
    self._model = model
    self._batch_hparams = batch_hparams

  def __call__(
      self, state: federated_algorithm.ServerState, round_num: int,
      train_clients: Sequence[Tuple[federated_data.ClientId,
                                    client_datasets.ClientDataset, Any]]
  ) -> Mapping[str, jnp.ndarray]:
    del round_num
    params = state.params
    batches = client_datasets.padded_batch_client_datasets(
        [client_dataset for _, client_dataset, _ in train_clients],
        self._batch_hparams)
    return models.evaluate_model(self._model, params, batches)


class FederatedExperimentConfig(NamedTuple):
  """Common configurations of a federated experiment.

  Attribues:
    root_dir: Root directory for experiment outputs (e.g. metrics).
    num_rounds: Number of federated training rounds.
    checkpoint_frequency: Checkpoint frequency in rounds. If <= 0, no
    checkpointing is done.
    num_checkpoints_to_keep: Maximum number of checkpoints to keep.
    eval_frequency: Evaluation frequency in rounds. If <= 0, no evaluation is
    done.
  """
  root_dir: str
  num_rounds: int
  checkpoint_frequency: int = 0
  num_checkpoints_to_keep: int = 1
  eval_frequency: int = 0


def run_federated_experiment(
    algorithm: federated_algorithm.FederatedAlgorithm,
    init_state: federated_algorithm.ServerState,
    client_sampler: client_samplers.ClientSampler,
    config: FederatedExperimentConfig,
    periodic_eval_fn_map: Optional[Mapping[str, Any]] = None,
    final_eval_fn_map: Optional[Mapping[str, EvaluationFn]] = None
) -> federated_algorithm.ServerState:
  """Runs the training loop of a federated algorithm experiment.

  Args:
    algorithm: Federated algorithm to use.
    init_state: Initial server state.
    client_sampler: Sampler for training clients.
    config: FederatedExperimentConfig configurations.
    periodic_eval_fn_map: Mapping of name to evaluation functions that are run
      repeatedly over multiple federated training rounds. The frequency is
      defined in `_FederatedExperimentConfig.eval_frequency`.
    final_eval_fn_map: Mapping of name to evaluation functions that are run at
      the very end of federated training. Typically, full test evaluation
      functions will be set here.

  Returns:
    Final state of the input federated algortihm after training.
  """
  if config.root_dir:
    tf.io.gfile.makedirs(config.root_dir)

  if periodic_eval_fn_map is None:
    periodic_eval_fn_map = {}
  if final_eval_fn_map is None:
    final_eval_fn_map = {}

  logger = fedjax_logging.Logger(config.root_dir)

  latest = checkpoint.load_latest_checkpoint(config.root_dir)
  if latest:
    state, last_round_num = latest
    start_round_num = last_round_num + 1
  else:
    state = init_state
    start_round_num = 1
  client_sampler.set_round_num(start_round_num)

  start = time.time()
  for round_num in range(start_round_num, config.num_rounds + 1):
    # Get a random state and randomly sample clients.
    clients = client_sampler.sample()
    client_ids = [i[0] for i in clients]
    logging.info('round_num %d: client_ids = %s', round_num, client_ids)

    # Run one round of the algorithm, where bulk of the work happens.
    state, _ = algorithm.apply(state, clients)

    # Save checkpoint.
    should_save_checkpoint = config.checkpoint_frequency and (
        round_num == start_round_num or
        round_num % config.checkpoint_frequency == 0)
    if should_save_checkpoint:
      checkpoint.save_checkpoint(config.root_dir, state, round_num,
                                 config.num_checkpoints_to_keep)

    # Run evaluation.
    should_run_eval = config.eval_frequency and (
        round_num == start_round_num or round_num % config.eval_frequency == 0)
    if should_run_eval:
      start_periodic_eval = time.time()
      for eval_name, eval_fn in periodic_eval_fn_map.items():
        if isinstance(eval_fn, EvaluationFn):
          metrics = eval_fn(state, round_num)
        elif isinstance(eval_fn, TrainClientsEvaluationFn):
          metrics = eval_fn(state, round_num, clients)
        else:
          raise ValueError(f'Invalid eval_fn type {type(eval_fn)}')
        if metrics:
          for metric_name, metric_value in metrics.items():
            logger.log(eval_name, metric_name, metric_value, round_num)
      logger.log('.', 'periodic_eval_duration_sec',
                 time.time() - start_periodic_eval, round_num)

    # Log the time it takes per round. Rough approximation since we're not
    # using DeviceArray.block_until_ready()
    logger.log('.', 'mean_round_duration_sec',
               (time.time() - start) / (round_num + 1 - start_round_num),
               round_num)

  # Block until previous work has finished.
  jnp.zeros([]).block_until_ready()

  # Logging overall time it took.
  num_rounds = config.num_rounds - start_round_num + 1
  mean_round_duration = ((time.time() - start) /
                         num_rounds if num_rounds > 0 else 0)

  # Final evaluation.
  final_eval_start = time.time()
  for eval_name, eval_fn in final_eval_fn_map.items():
    metrics = eval_fn(state, round_num)
    if metrics:
      metrics_path = os.path.join(config.root_dir, f'{eval_name}.tsv')
      with tf.io.gfile.GFile(metrics_path, 'w') as f:
        f.write('\t'.join(metrics.keys()) + '\n')
        f.write('\t'.join([str(v) for v in metrics.values()]))
  # DeviceArray.block_until_ready() isn't needed here since we write to file.
  final_eval_duration = time.time() - final_eval_start
  logging.info('mean_round_duration = %f sec.', mean_round_duration)
  logging.info('final_eval_duration = %f sec.', final_eval_duration)
  return state
