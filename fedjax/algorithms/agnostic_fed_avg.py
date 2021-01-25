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
"""Agnostic federated averaging implementation.

Based on the paper:

Agnostic Federated Learning
    Mehryar Mohri, Gary Sivek, Ananda Theertha Suresh
    https://arxiv.org/abs/1902.00146

This implementation only supports the horizontal partitions case in which each
client belongs to only a single domain (e.g. user locale).
"""

import enum
from typing import Callable, List, NamedTuple, Tuple

from fedjax import core
import jax.numpy as jnp


class AgnosticFedAvgState(NamedTuple):
  """State of server for AgnosticFedAvg passed between rounds.

  Attributes:
    params: A pytree representing the server model parameters.
    server_opt_state: A pytree representing the server optimizer state.
    domain_weights: Weights per domain applied to weight in weighted average.
  """
  params: core.Params
  server_opt_state: core.OptState
  domain_weights: jnp.ndarray


# TODO(b/162840323): Implement SGD.
@enum.unique
class DomainAlgorithm(enum.Enum):
  """Domain weight update algorithm options.

  Attributes:
    EG: Exponentiated gradient update.
    NONE: No update.
  """
  EG = 'EG'
  NONE = 'NONE'


class AgnosticFedAvgHParams(NamedTuple):
  """Hyperparameters for agnostic federated averaging.

  Attributes:
    train_data_hparams: Hyperparameters for training client data preparation.
    init_domain_weights: Initial weights per domain that sums to 1.
    domain_id_fn: Given client id returns the domain id for that client.
    domain_learning_rate: Learning rate for domain weight update.
    domain_algorithm: Algorithm used to update domain weights each round.
  """
  train_data_hparams: core.ClientDataHParams
  init_domain_weights: Tuple[float, ...]
  domain_id_fn: Callable[[str], int] = lambda _: 0
  domain_learning_rate: float = 0.1
  domain_algorithm: DomainAlgorithm = DomainAlgorithm.EG


def _update_domain_weights(domain_weights: jnp.ndarray,
                           domain_loss: jnp.ndarray,
                           domain_learning_rate: float,
                           domain_algorithm: DomainAlgorithm) -> jnp.ndarray:
  """Updates domain weights following input algorithm.

  Args:
    domain_weights: Weights per domain that sums to 1.
    domain_loss: Average loss per domain.
    domain_learning_rate: Learning rate for domain update.
    domain_algorithm: Domain weight update strategy.

  Returns:
    Updated domain weights.

  Raises:
    ValueError: If an unsupported DomainAlgorithm is specified.
  """
  if domain_algorithm == DomainAlgorithm.EG:
    new_domain_weights = domain_weights * jnp.exp(
        domain_learning_rate * domain_loss)
    new_domain_weights = jnp.maximum(new_domain_weights,
                                     jnp.zeros_like(new_domain_weights))
    return new_domain_weights / jnp.sum(new_domain_weights)
  elif domain_algorithm == DomainAlgorithm.NONE:
    return domain_weights
  else:
    raise ValueError(f'Unsupported domain algorithm {domain_algorithm}.')


class AgnosticFedAvg(core.FederatedAlgorithm):
  """Agnostic federated averaging algorithm."""

  def __init__(self, federated_data: core.FederatedData, model: core.Model,
               client_optimizer: core.Optimizer,
               server_optimizer: core.Optimizer, hparams: AgnosticFedAvgHParams,
               rng_seq: core.PRNGSequence):
    """Initializes HypCluster algorithm.

    Args:
      federated_data: Federated data separated per client.
      model: Model implementation.
      client_optimizer: Client optimizer.
      server_optimizer: Server optimizer.
      hparams: Hyperparameters for agnostic federated averaging.
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

  def init_state(self) -> AgnosticFedAvgState:
    params = self._model.init_params(next(self._rng_seq))
    server_opt_state = self._server_optimizer.init_fn(params)
    domain_weights = jnp.array(self._hparams.init_domain_weights)
    return AgnosticFedAvgState(
        params=params,
        server_opt_state=server_opt_state,
        domain_weights=domain_weights)

  def run_round(self, state: AgnosticFedAvgState,
                client_ids: List[str]) -> AgnosticFedAvgState:
    """Runs one round of agnostic federated averaging."""
    client_metrics = core.evaluate_multiple_clients(
        federated_data=self.federated_data,
        client_ids=client_ids,
        model=self.model,
        params=state.params,
        client_data_hparams=self._hparams.train_data_hparams)

    domain_ids = []
    num_domains = len(state.domain_weights)
    domain_loss = [0.0] * num_domains
    domain_num = [0.0] * num_domains
    for client_id, metrics in zip(client_ids, client_metrics):
      domain_id = self._hparams.domain_id_fn(client_id)
      domain_ids.append(domain_id)
      if metrics:
        domain_loss[domain_id] += metrics['loss'] * metrics['num_examples']
        domain_num[domain_id] += metrics['num_examples']
    domain_loss = jnp.array(domain_loss)
    domain_num = jnp.array(domain_num)

    # Train model per client.
    client_states = core.train_multiple_clients(
        federated_data=self.federated_data,
        client_ids=client_ids,
        client_trainer=self._client_trainer,
        init_client_trainer_state=self._client_trainer.init_state(state.params),
        rng_seq=self._rng_seq,
        client_data_hparams=self._hparams.train_data_hparams)

    # Scale domain weights by the proportion of total domain examples.
    scaled_dw = state.domain_weights / domain_num
    scaled_dw = jnp.where(domain_num == 0., 0., scaled_dw)

    # Weighted average of param delta across clients.
    def select_delta_params_and_weight(element):
      idx, client_state = element
      delta_params = core.tree_multimap(lambda a, b: a - b, state.params,
                                        client_state.params)
      # Use client weight scaled by domain weight weight for weighted average.
      domain_id = domain_ids[idx]
      client_weight = scaled_dw[domain_id] * client_state.num_examples
      return delta_params, client_weight

    delta_params_and_weight = map(select_delta_params_and_weight,
                                  enumerate(client_states))
    delta_params = core.tree_mean(delta_params_and_weight)

    # Server update.
    avg_loss_per_domain = jnp.where(domain_num == 0., 0.,
                                    domain_loss / domain_num)
    domain_weights = _update_domain_weights(
        domain_weights=state.domain_weights,
        domain_loss=avg_loss_per_domain,
        domain_learning_rate=self._hparams.domain_learning_rate,
        domain_algorithm=self._hparams.domain_algorithm)
    updates, server_opt_state = self._server_optimizer.update_fn(
        delta_params, state.server_opt_state)
    params = self._server_optimizer.apply_updates(state.params, updates)
    return AgnosticFedAvgState(
        params=params,
        server_opt_state=server_opt_state,
        domain_weights=domain_weights)
