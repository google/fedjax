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

from typing import Any, Callable, Iterable, Mapping, Tuple

from fedjax.core import dataclasses
from fedjax.core import tree_util
from fedjax.core.typing import Params
from fedjax.core.typing import PRNGKey
from fedjax.experimental import client_datasets
from fedjax.experimental import federated_algorithm
from fedjax.experimental import federated_data
from fedjax.experimental import optimizers
from fedjax.experimental.typing import BatchExample

import jax
import jax.numpy as jnp

Grads = Params


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
  """Builds the basic implementation of federated averaging."""

  def init(params: Params) -> ServerState:
    opt_state = server_optimizer.init(params)
    return ServerState(params, opt_state)

  def apply(
      server_state: ServerState,
      clients: Iterable[Tuple[federated_data.ClientId,
                              client_datasets.ClientDataset, PRNGKey]]
  ) -> Tuple[ServerState, Mapping[federated_data.ClientId, Any]]:
    client_diagnostics = {}
    client_outputs = []
    for client_id, client_dataset, client_rng in clients:
      delta_params, num_examples, step_results = client_update(
          server_state.params, client_dataset, client_rng)
      client_outputs.append((delta_params, num_examples))
      client_diagnostics[client_id] = step_results
    server_state = server_update(server_state, client_outputs)
    return server_state, client_diagnostics

  def client_update(server_params, client_dataset, client_rng):
    params = server_params
    opt_state = client_optimizer.init(params)
    num_examples = 0.
    step_results = []
    for batch in client_dataset.shuffle_repeat_batch(client_dataset_hparams):
      client_rng, use_rng = jax.random.split(client_rng)
      grads = grad_fn(params, batch, use_rng)
      opt_state, params = client_optimizer.apply(grads, opt_state, params)
      num_examples += len(next(iter(batch.values())))
      # We record the l2 norm of gradients as an example, but it is not required
      # for the algorithm.
      step_results.append({
          'grad_l2_norm':
              sum(jnp.vdot(x, x) for x in jax.tree_util.tree_leaves(grads))
      })
    delta_params = jax.tree_util.tree_multimap(lambda a, b: a - b,
                                               server_params, params)
    return delta_params, num_examples, step_results

  def server_update(server_state, client_outputs):
    delta_params = tree_util.tree_mean(client_outputs)
    opt_state, params = server_optimizer.apply(delta_params,
                                               server_state.opt_state,
                                               server_state.params)
    return ServerState(params, opt_state)

  return federated_algorithm.FederatedAlgorithm(init, apply)
