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
from fedjax.experimental.typing import BatchExample

import jax
import jax.numpy as jnp

Grads = Params


def _build_for_each_client_fns(grad_fn, client_optimizer):
  """Builds client_init, client_step, client_final for for_each_client."""

  def client_init(server_params, client_rng):
    opt_state = client_optimizer.init(server_params)
    client_step_state = {
        'params': server_params,
        'opt_state': opt_state,
        'rng': client_rng,
        'num_examples': 0.,
    }
    return client_step_state

  def client_step(client_step_state, batch):
    params = client_step_state['params']
    opt_state = client_step_state['opt_state']
    rng, use_rng = jax.random.split(client_step_state['rng'])
    grads = grad_fn(params, batch, use_rng)
    opt_state, params = client_optimizer.apply(grads, opt_state, params)
    num_examples = client_datasets.num_examples(batch)
    client_step_state = {
        'params': params,
        'opt_state': opt_state,
        'rng': rng,
        'num_examples': client_step_state['num_examples'] + num_examples,
    }
    # We record the l2 norm of gradients as an example, but it is not required
    # for the algorithm.
    client_step_result = {
        'grad_l2_norm':
            sum(jnp.vdot(x, x) for x in jax.tree_util.tree_leaves(grads))
    }
    return client_step_state, client_step_result

  def client_final(server_params, client_step_state):
    delta_params = jax.tree_util.tree_multimap(lambda a, b: a - b,
                                               server_params,
                                               client_step_state['params'])
    client_output = (delta_params, client_step_state['num_examples'])
    return client_output

  return client_init, client_step, client_final


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

  client_init, client_step, client_final = _build_for_each_client_fns(
      grad_fn, client_optimizer)
  for_each_client_ = for_each_client.for_each_client(client_init, client_step,
                                                     client_final)

  def init(params: Params) -> ServerState:
    opt_state = server_optimizer.init(params)
    return ServerState(params, opt_state)

  def apply(
      server_state: ServerState,
      clients: Iterable[Tuple[federated_data.ClientId,
                              client_datasets.ClientDataset, PRNGKey]]
  ) -> Tuple[ServerState, Mapping[federated_data.ClientId, Any]]:
    batched_clients = map(
        lambda c:
        (c[0], c[1].shuffle_repeat_batch(client_dataset_hparams), c[2]),
        clients)
    server_params = server_state.params
    client_diagnostics = {}

    # We need to use this work around to split off and isolate client outputs
    # from the client step results without storing all of the outputs of the
    # generator returned by for_each_client_() in memory.
    def client_output_generator():
      for client_id, client_output, client_step_results in for_each_client_(
          batched_clients, server_params):
        client_diagnostics[client_id] = client_step_results
        yield client_output

    client_outputs = client_output_generator()
    server_state = server_update(server_state, client_outputs)
    return server_state, client_diagnostics

  def server_update(server_state, client_outputs):
    delta_params = tree_util.tree_mean(client_outputs)
    opt_state, params = server_optimizer.apply(delta_params,
                                               server_state.opt_state,
                                               server_state.params)
    return ServerState(params, opt_state)

  return federated_algorithm.FederatedAlgorithm(init, apply)
