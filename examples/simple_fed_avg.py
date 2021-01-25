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
"""Simple federated averaging implementation.

Identical to the implementation at //fedjax/algorithms/fed_avg.py

Based on the paper:

Communication-Efficient Learning of Deep Networks from Decentralized Data
    H. Brendan McMahan, Eider Moore, Daniel Ramage,
    Seth Hampson, Blaise Aguera y Arcas. AISTATS 2017.
    https://arxiv.org/abs/1602.05629
"""

from typing import List, NamedTuple

import fedjax


class SimpleFedAvgState(NamedTuple):
  """State of server passed between rounds.

  Attributes:
    params: A pytree representing the server model parameters.
    server_opt_state: A pytree representing the server optimizer state.
  """
  params: fedjax.Params
  server_opt_state: fedjax.OptState


class SimpleFedAvgHParams(NamedTuple):
  """Hyperparameters for federated averaging.

  Attributes:
    train_data_hparams: Hyperparameters for training data preparation.
  """
  train_data_hparams: fedjax.ClientDataHParams


class SimpleFedAvg(fedjax.FederatedAlgorithm):
  """Simple federated averaging algorithm."""

  def __init__(self, federated_data: fedjax.FederatedData, model: fedjax.Model,
               client_optimizer: fedjax.Optimizer,
               server_optimizer: fedjax.Optimizer, hparams: SimpleFedAvgHParams,
               rng_seq: fedjax.PRNGSequence):
    """Initializes SimpleFedAvg algorithm.

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
    self._client_trainer = fedjax.DefaultClientTrainer(model, client_optimizer)

  @property
  def federated_data(self) -> fedjax.FederatedData:
    return self._federated_data

  @property
  def model(self) -> fedjax.Model:
    return self._model

  def init_state(self) -> SimpleFedAvgState:
    params = self._model.init_params(next(self._rng_seq))
    server_opt_state = self._server_optimizer.init_fn(params)
    return SimpleFedAvgState(params=params, server_opt_state=server_opt_state)

  def run_round(self, state: SimpleFedAvgState,
                client_ids: List[str]) -> SimpleFedAvgState:
    """Runs one round of federated averaging."""
    # Train model per client.
    client_outputs = fedjax.train_multiple_clients(
        federated_data=self.federated_data,
        client_ids=client_ids,
        client_trainer=self._client_trainer,
        init_client_trainer_state=self._client_trainer.init_state(state.params),
        rng_seq=self._rng_seq,
        client_data_hparams=self._hparams.train_data_hparams)

    # Weighted average of param delta across clients.
    def get_delta_params_and_weight(client_output):
      delta_params = fedjax.tree_multimap(lambda a, b: a - b, state.params,
                                          client_output.params)
      return delta_params, client_output.num_examples

    delta_params_and_weight = map(get_delta_params_and_weight, client_outputs)
    delta_params = fedjax.tree_mean(delta_params_and_weight)

    # Server state update.
    updates, server_opt_state = self._server_optimizer.update_fn(
        delta_params, state.server_opt_state)
    params = self._server_optimizer.apply_updates(state.params, updates)
    return SimpleFedAvgState(params=params, server_opt_state=server_opt_state)
