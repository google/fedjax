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
"""Mime Lite implementation.

Based on the paper:

Mime: Mimicking Centralized Stochastic Algorithms in Federated Learning
    Sai Praneeth Karimireddy, Martin Jaggi, Satyen Kale, Mehryar Mohri,
    Sashank J. Reddi, Sebastian U. Stich, Ananda Theertha Suresh
    https://arxiv.org/abs/2008.03606
"""

import functools
from typing import Iterable, List, NamedTuple

from fedjax import core
import jax
import jax.numpy as jnp


class MimeLiteState(NamedTuple):
  """State of server for Mime/MimeLite passed between rounds.

  Attributes:
    params: A pytree representing the server model parameters.
    opt_state: A pytree representing the base optimizer state.
  """
  params: core.Params
  opt_state: core.OptState


class MimeLiteHParams(NamedTuple):
  """Hyperparameters for Mime Lite algorithm.

  Attributes:
    train_data_hparams: Hyperparameters for training client data preparation.
    combined_data_hparams: Hyperparameters for training data preparation.
    server_learning_rate: Server learning rate scales the update from clients
      before applying to server. Defaults to 1.
  """
  train_data_hparams: core.ClientDataHParams
  combined_data_hparams: core.ClientDataHParams
  server_learning_rate: float = 1.0


class MimeLiteClientTrainer(core.DefaultClientTrainer):
  """Single client trainer that doesn't update optimizer state."""

  @functools.partial(jax.jit, static_argnums=0)
  def one_step(self, client_trainer_state: core.DefaultClientTrainerState,
               batch: core.Batch,
               rng: core.PRNGKey) -> core.DefaultClientTrainerState:
    backward_pass_output = self._model.backward_pass(
        client_trainer_state.params, batch, rng)
    params_updates, _ = self._optimizer.update_fn(
        backward_pass_output.grads, client_trainer_state.opt_state)
    params = self._optimizer.apply_updates(client_trainer_state.params,
                                           params_updates)
    return core.DefaultClientTrainerState(
        params=params,
        opt_state=client_trainer_state.opt_state,
        num_examples=client_trainer_state.num_examples +
        backward_pass_output.num_examples)


def compute_gradient(stream: Iterable[core.Batch], params: core.Params,
                     model: core.Model,
                     rng_seq: core.PRNGSequence) -> core.Updates:
  """Computes the gradient over the full stream at params."""
  num_examples = 0.
  grads_sum = jax.tree_map(jnp.zeros_like, params)
  for batch, rng in zip(stream, rng_seq):
    backward_pass_output = model.backward_pass(params, batch, rng)
    grads = backward_pass_output.grads
    batch_num_examples = backward_pass_output.num_examples
    grads_sum = jax.tree_multimap(
        lambda p, q, bs=batch_num_examples: p * bs + q, grads, grads_sum)
    num_examples += batch_num_examples
  return jax.tree_map(lambda p: p / num_examples, grads_sum)


class MimeLite(core.FederatedAlgorithm):
  """Mime Lite algorithm."""

  def __init__(self, federated_data: core.FederatedData, model: core.Model,
               base_optimizer: core.Optimizer, hparams: MimeLiteHParams,
               rng_seq: core.PRNGSequence):
    """Initializes MimeLite algorithm.

    Args:
      federated_data: Federated data separated per client.
      model: Model implementation.
      base_optimizer: Base centralized optimizer which we will try mimic.
      hparams: Hyperparameters for federated averaging.
      rng_seq: Iterator of JAX random keys.
    """
    self._federated_data = federated_data
    self._model = model
    self._base_optimizer = base_optimizer
    self._hparams = hparams
    self._rng_seq = rng_seq
    self._client_trainer = MimeLiteClientTrainer(model, base_optimizer)

  @property
  def federated_data(self) -> core.FederatedData:
    return self._federated_data

  @property
  def model(self) -> core.Model:
    return self._model

  def init_state(self) -> MimeLiteState:
    params = self._model.init_params(next(self._rng_seq))
    opt_state = self._base_optimizer.init_fn(params)
    return MimeLiteState(params=params, opt_state=opt_state)

  def run_round(self, state: MimeLiteState,
                client_ids: List[str]) -> MimeLiteState:
    """Runs one round of MimeLite."""
    # Train model per client.
    client_states = core.train_multiple_clients(
        federated_data=self.federated_data,
        client_ids=client_ids,
        client_trainer=self._client_trainer,
        init_client_trainer_state=self._client_trainer.init_state(
            state.params, state.opt_state),
        rng_seq=self._rng_seq,
        client_data_hparams=self._hparams.train_data_hparams)

    # Weighted average of param delta across clients.
    def select_delta_params_and_weight(client_state):
      delta_params = core.tree_multimap(lambda a, b: a - b, state.params,
                                        client_state.params)
      return delta_params, client_state.num_examples

    delta_params_and_weight = map(select_delta_params_and_weight, client_states)
    delta_params = core.tree_mean(delta_params_and_weight)

    # Compute full-batch gradient at server params on train data.
    combined_dataset = core.create_tf_dataset_for_clients(
        self.federated_data, client_ids)
    combined_dataset = core.preprocess_tf_dataset(
        combined_dataset, self._hparams.combined_data_hparams)
    server_grads = compute_gradient(
        stream=combined_dataset.as_numpy_iterator(),
        params=state.params,
        model=self._model,
        rng_seq=self._rng_seq,
    )

    # Server params uses weighted average of client updates, scaled by the
    # server_learning_rate.
    params = core.tree_multimap(
        lambda p, q: p - self._hparams.server_learning_rate * q, state.params,
        delta_params)
    # Update server opt_state using base_optimizer and server gradient.
    _, opt_state = self._base_optimizer.update_fn(server_grads, state.opt_state)
    return MimeLiteState(params=params, opt_state=opt_state)
