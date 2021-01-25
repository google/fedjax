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
"""Federated averaging implementation.

Based on the paper:

Communication-Efficient Learning of Deep Networks from Decentralized Data
    H. Brendan McMahan, Eider Moore, Daniel Ramage,
    Seth Hampson, Blaise Aguera y Arcas. AISTATS 2017.
    https://arxiv.org/abs/1602.05629
"""

from typing import List, NamedTuple

from fedjax import core


class FedAvgState(NamedTuple):
  """State of server for FedAvg passed between rounds.

  Attributes:
    params: A pytree representing the server model parameters.
    server_opt_state: A pytree representing the server optimizer state.
  """
  params: core.Params
  server_opt_state: core.OptState


class FedAvgHParams(NamedTuple):
  """Hyperparameters for federated averaging.

  Attributes:
    train_data_hparams: Hyperparameters for training client data preparation.
  """
  train_data_hparams: core.ClientDataHParams


class FedAvg(core.FederatedAlgorithm):
  """Federated averaging algorithm."""

  def __init__(self, federated_data: core.FederatedData, model: core.Model,
               client_optimizer: core.Optimizer,
               server_optimizer: core.Optimizer, hparams: FedAvgHParams,
               rng_seq: core.PRNGSequence):
    """Initializes FedAvg algorithm.

    Args:
      federated_data: Federated data separated per client.
      model: Model implementation.
      client_optimizer: Client optimizer.
      server_optimizer: Server optimizer.
      hparams: Hyperparameters for federated averaging.
      rng_seq: Iterator of JAX random keys.
    """
    self._federated_data = federated_data
    self._model = model
    self._client_optimizer = client_optimizer
    self._server_optimizer = server_optimizer
    self._hparams = hparams
    self._rng_seq = rng_seq
    self._client_trainer = core.DefaultClientTrainer(model, client_optimizer)

  @property
  def federated_data(self) -> core.FederatedData:
    return self._federated_data

  @property
  def model(self) -> core.Model:
    return self._model

  def init_state(self) -> FedAvgState:
    params = self._model.init_params(next(self._rng_seq))
    server_opt_state = self._server_optimizer.init_fn(params)
    return FedAvgState(params=params, server_opt_state=server_opt_state)

  def run_round(self, state: FedAvgState, client_ids: List[str]) -> FedAvgState:
    """Runs one round of federated averaging."""
    # Train model per client.
    client_states = core.train_multiple_clients(
        federated_data=self.federated_data,
        client_ids=client_ids,
        client_trainer=self._client_trainer,
        init_client_trainer_state=self._client_trainer.init_state(state.params),
        rng_seq=self._rng_seq,
        client_data_hparams=self._hparams.train_data_hparams)

    # Weighted average of param delta across clients.
    def select_delta_params_and_weight(client_state):
      delta_params = core.tree_multimap(lambda a, b: a - b, state.params,
                                        client_state.params)
      return delta_params, client_state.num_examples

    delta_params_and_weight = map(select_delta_params_and_weight, client_states)
    delta_params = core.tree_mean(delta_params_and_weight)

    # Server state update.
    updates, server_opt_state = self._server_optimizer.update_fn(
        delta_params, state.server_opt_state)
    params = self._server_optimizer.apply_updates(state.params, updates)
    return FedAvgState(params=params, server_opt_state=server_opt_state)
