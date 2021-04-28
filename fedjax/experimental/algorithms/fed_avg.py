# Copyright 2021 Google LLC
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
"""Federated averaging implementation using fedjax.experimental.

This is the more performant implementation that matches what would be used in
the fedjax.algorithms.fed_avg. The key difference between this and the basic
implementation is the usage of fedjax.experimental.for_each_client.

Based on the paper:

Communication-Efficient Learning of Deep Networks from Decentralized Data
    H. Brendan McMahan, Eider Moore, Daniel Ramage,
    Seth Hampson, Blaise Aguera y Arcas. AISTATS 2017.
    https://arxiv.org/abs/1602.05629
"""

from typing import Any, Callable, Iterable, Mapping, Tuple

from fedjax.core import dataclasses
from fedjax.core import tree_util
from fedjax.core.typing import Params
from fedjax.core.typing import PRNGKey
from fedjax.experimental import client_datasets
from fedjax.experimental import federated_algorithm
from fedjax.experimental import federated_data
from fedjax.experimental import for_each_client
from fedjax.experimental import optimizers
from fedjax.experimental import tree_util as exp_tree_util
from fedjax.experimental.typing import BatchExample

import jax

Grads = Params


def create_train_for_each_client(grad_fn, client_optimizer):
  """Builds client_init, client_step, client_final for for_each_client."""

  def client_init(server_params, client_rng):
    opt_state = client_optimizer.init(server_params)
    client_step_state = {
        'params': server_params,
        'opt_state': opt_state,
        'rng': client_rng,
    }
    return client_step_state

  def client_step(client_step_state, batch):
    rng, use_rng = jax.random.split(client_step_state['rng'])
    grads = grad_fn(client_step_state['params'], batch, use_rng)
    opt_state, params = client_optimizer.apply(grads,
                                               client_step_state['opt_state'],
                                               client_step_state['params'])
    next_client_step_state = {
        'params': params,
        'opt_state': opt_state,
        'rng': rng,
    }
    return next_client_step_state

  def client_final(server_params, client_step_state):
    delta_params = jax.tree_util.tree_multimap(lambda a, b: a - b,
                                               server_params,
                                               client_step_state['params'])
    return delta_params

  return for_each_client.for_each_client(client_init, client_step, client_final)


@dataclasses.dataclass
class ServerState:
  """State of server passed between rounds.

  Attributes:
    params: A pytree representing the server model parameters.
    opt_state: A pytree representing the server optimizer state.
  """
  params: Params
  opt_state: optimizers.OptState


def federated_averaging(
    grad_fn: Callable[[Params, BatchExample, PRNGKey], Grads],
    client_optimizer: optimizers.Optimizer,
    server_optimizer: optimizers.Optimizer,
    client_dataset_hparams: client_datasets.ShuffleRepeatBatchHParams
) -> federated_algorithm.FederatedAlgorithm:
  """Builds federated averaging."""

  train_for_each_client = create_train_for_each_client(grad_fn,
                                                       client_optimizer)

  def init(params: Params) -> ServerState:
    opt_state = server_optimizer.init(params)
    return ServerState(params, opt_state)

  def apply(
      server_state: ServerState,
      clients: Iterable[Tuple[federated_data.ClientId,
                              client_datasets.ClientDataset, PRNGKey]]
  ) -> Tuple[ServerState, Mapping[federated_data.ClientId, Any]]:
    client_num_examples = {}
    client_diagnostics = {}

    def batch_clients():
      for client_id, client_dataset, client_rng in clients:
        client_num_examples[client_id] = len(client_dataset)
        yield (client_id,
               client_dataset.shuffle_repeat_batch(client_dataset_hparams),
               client_rng)

    # We need to use this work around to split off and isolate client outputs
    # without storing all of the outputs in memory.
    def client_deltas_weights_generator():
      for client_id, client_delta_params in train_for_each_client(
          server_state.params, batch_clients()):
        # We record the l2 norm of client updates as an example, but it is not
        # required for the algorithm.
        client_diagnostics[client_id] = {
            'delta_l2_norm': exp_tree_util.tree_l2_norm(client_delta_params)
        }
        yield client_delta_params, client_num_examples[client_id]

    client_deltas_weights = client_deltas_weights_generator()
    server_state = server_update(server_state, client_deltas_weights)
    return server_state, client_diagnostics

  def server_update(server_state, client_deltas_weights):
    delta_params = tree_util.tree_mean(client_deltas_weights)
    opt_state, params = server_optimizer.apply(delta_params,
                                               server_state.opt_state,
                                               server_state.params)
    return ServerState(params, opt_state)

  return federated_algorithm.FederatedAlgorithm(init, apply)
