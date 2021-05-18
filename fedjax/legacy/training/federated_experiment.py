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

import os.path
import time
from typing import Any, Callable, List, Mapping, NamedTuple, Optional, TypeVar

from absl import logging
from fedjax.legacy import core
from fedjax.legacy.training import checkpoint
from fedjax.legacy.training import logging as fedjax_logging
import jax.numpy as jnp
import numpy as np
import tensorflow as tf


def set_tf_cpu_only():
  """Restricts TensorFlow device visibility to only CPU.

  TensorFlow is only used for data loading, so we prevent it from allocating
  GPU/TPU memory.
  """
  tf.config.experimental.set_visible_devices([], 'GPU')
  tf.config.experimental.set_visible_devices([], 'TPU')


def get_pseudo_random_state(round_num: int,
                            random_seed: Optional[int] = None
                           ) -> np.random.RandomState:
  """Computes a numpy random state as a function of round_num and random_seed."""
  #  Settings for a multiplicative linear congruential generator (aka Lehmer
  #  generator) suggested in 'Random Number Generators: Good
  #  Ones are Hard to Find' by Park and Miller.
  if isinstance(random_seed, int):
    mlcg_modulus = 2**(31) - 1
    mlcg_multiplier = 16807
    mlcg_start = np.random.RandomState(random_seed).randint(1, mlcg_modulus - 1)
    return np.random.RandomState(
        pow(mlcg_multiplier, round_num, mlcg_modulus) * mlcg_start %
        mlcg_modulus)
  return np.random.RandomState()


class FederatedExperimentConfig(NamedTuple):
  root_dir: str
  num_rounds: int
  num_clients_per_round: int
  sample_client_random_seed: Optional[int] = None
  checkpoint_frequency: int = 0
  num_checkpoints_to_keep: int = 1
  eval_frequency: int = 0


class ClientEvaluationFn:
  """Evaluation function for a subset of client(s)."""

  def __init__(self, federated_data: core.FederatedData, model: core.Model,
               config: FederatedExperimentConfig):
    self._federated_data = federated_data
    self._model = model
    self._num_clients_per_round = config.num_clients_per_round
    self._sample_client_random_seed = config.sample_client_random_seed

  def _sample_clients(self, round_num: int) -> List[str]:
    random_state = get_pseudo_random_state(round_num,
                                           self._sample_client_random_seed)
    return list(
        random_state.choice(
            self._federated_data.client_ids,
            size=self._num_clients_per_round,
            replace=False))

  def __call__(self, state: Any, round_num: int) -> core.MetricResults:
    client_ids = self._sample_clients(round_num)
    combined_dataset = core.create_tf_dataset_for_clients(
        self._federated_data, client_ids=client_ids)
    return core.evaluate_single_client(combined_dataset, self._model,
                                       state.params)


class FullEvaluationFn:
  """Evaluation function for all of client(s)."""

  def __init__(self, federated_data: core.FederatedData, model: core.Model):
    self._dataset = core.dataset_util.create_tf_dataset_for_clients(
        federated_data)
    self._model = model

  def __call__(self, state: Any, round_num: int) -> core.MetricResults:
    del round_num
    return core.evaluate_single_client(self._dataset, self._model, state.params)


def _block_until_ready_state(state: Any):
  """Recursively calls block_until_ready on valid pytree leaves of state."""

  def _safe_block_until_ready(l):
    if isinstance(l, jnp.ndarray):
      l.block_until_ready()

  core.tree_map(_safe_block_until_ready, state)


T = TypeVar('T')


# TODO(theertha): Add functionality for warmstart / custom initialization.
def run_federated_experiment(
    config: FederatedExperimentConfig,
    federated_algorithm: core.FederatedAlgorithm[T],
    periodic_eval_fn_map: Optional[Mapping[str, Callable[
        [T, int], core.MetricResults]]] = None,
    final_eval_fn_map: Optional[Mapping[str,
                                        Callable[[T, int],
                                                 core.MetricResults]]] = None
) -> T:
  """Runs federated algorithm experiment and auxiliary processes.

  Args:
    config: FederatedExperimentConfig configurations.
    federated_algorithm: FederatedAlgorithm to be run over multiple rounds.
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
    state, start_round_num = latest
  else:
    state = federated_algorithm.init_state()
    start_round_num = 1

  start = time.time()
  for round_num in range(start_round_num, config.num_rounds + 1):
    # Get a random state and randomly sample clients.
    random_state = get_pseudo_random_state(round_num,
                                           config.sample_client_random_seed)
    client_ids = list(
        random_state.choice(
            federated_algorithm.federated_data.client_ids,
            size=config.num_clients_per_round,
            replace=False))
    logging.info('round_num %d: client_ids = %s', round_num, client_ids)

    # Run one round of the algorithm, where bulk of the work happens.
    state = federated_algorithm.run_round(state, client_ids)

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
        metrics = eval_fn(state, round_num)
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

  # Logging overall time it took.
  # DeviceArray.block_until_ready() is needed for accurate timing due to
  # https://jax.readthedocs.io/en/latest/async_dispatch.html.
  _block_until_ready_state(state)
  num_rounds = config.num_rounds - start_round_num
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
