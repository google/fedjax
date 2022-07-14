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
"""Basic federated averaging implementation using fedjax.experimental.

This is the basic implementation in that sense that we write out the native
Python for loops over clients by hand. While this is much more straightforward
to implement, it's not the fastest. This is a good starting point for
implementing custom algorithms since it tends to follow pseudocode closely.

Based on the paper:

Communication-Efficient Learning of Deep Networks from Decentralized Data
    H. Brendan McMahan, Eider Moore, Daniel Ramage,
    Seth Hampson, Blaise Aguera y Arcas. AISTATS 2017.
    https://arxiv.org/abs/1602.05629
"""

from typing import Any, Callable, Mapping, Sequence, Tuple

import fedjax

import jax

ClientId = bytes
Grads = fedjax.Params


@fedjax.dataclass
class ServerState:
  """State of server passed between rounds.

  Attributes:
    params: A pytree representing the server model parameters.
    opt_state: A pytree representing the server optimizer state.
  """
  params: fedjax.Params
  opt_state: fedjax.OptState


def federated_averaging(
    grad_fn: Callable[[fedjax.Params, fedjax.BatchExample, fedjax.PRNGKey],
                      Grads],
    client_optimizer: fedjax.Optimizer,
    server_optimizer: fedjax.Optimizer,
    client_batch_hparams: fedjax.ShuffleRepeatBatchHParams
) -> fedjax.FederatedAlgorithm:
  """Builds the basic implementation of federated averaging."""

  def init(params: fedjax.Params) -> ServerState:
    opt_state = server_optimizer.init(params)
    return ServerState(params, opt_state)

  def apply(
      server_state: ServerState,
      clients: Sequence[Tuple[ClientId, fedjax.ClientDataset, fedjax.PRNGKey]]
  ) -> Tuple[ServerState, Mapping[ClientId, Any]]:
    client_diagnostics = {}
    # We use a list here for clarity, but we strongly recommend avoiding loading
    # all client outputs into memory since the outputs can be quite large
    # depending on the size of the model.
    client_delta_params_weights = []
    for client_id, client_dataset, client_rng in clients:
      delta_params = client_update(server_state.params, client_dataset,
                                   client_rng)
      client_delta_params_weights.append((delta_params, len(client_dataset)))
      # We record the l2 norm of client updates as an example, but it is not
      # required for the algorithm.
      client_diagnostics[client_id] = {
          'delta_l2_norm': fedjax.tree_util.tree_l2_norm(delta_params)
      }
    mean_delta_params = fedjax.tree_util.tree_mean(client_delta_params_weights)
    server_state = server_update(server_state, mean_delta_params)
    return server_state, client_diagnostics

  def client_update(server_params, client_dataset, client_rng):
    params = server_params
    opt_state = client_optimizer.init(params)
    for batch in client_dataset.shuffle_repeat_batch(client_batch_hparams):
      client_rng, use_rng = jax.random.split(client_rng)
      grads = grad_fn(params, batch, use_rng)
      opt_state, params = client_optimizer.apply(grads, opt_state, params)
    delta_params = jax.tree_util.tree_map(lambda a, b: a - b,
                                               server_params, params)
    return delta_params

  def server_update(server_state, mean_delta_params):
    opt_state, params = server_optimizer.apply(mean_delta_params,
                                               server_state.opt_state,
                                               server_state.params)
    return ServerState(params, opt_state)

  return fedjax.FederatedAlgorithm(init, apply)
