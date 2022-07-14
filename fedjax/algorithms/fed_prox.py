# Copyright 2022 Google LLC
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
"""FedProx implementation using fedjax.core.

Federated Optimization in Heterogeneous Networks
    Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar,
    Virginia Smith. MLSys 2020.
    https://arxiv.org/abs/1812.06127
"""

from typing import Any, Callable, Mapping, Sequence, Tuple

from fedjax.algorithms import fed_avg

from fedjax.core import client_datasets
from fedjax.core import federated_algorithm
from fedjax.core import federated_data
from fedjax.core import for_each_client
from fedjax.core import optimizers
from fedjax.core import tree_util
from fedjax.core.typing import BatchExample
from fedjax.core.typing import Params
from fedjax.core.typing import PRNGKey

import jax
import jax.numpy as jnp

Grads = Params
ServerState = fed_avg.ServerState


def create_train_for_each_client(grad_fn, client_optimizer):
  """Builds client_init, client_step, client_final for for_each_client."""

  def client_init(server_params, client_rng):
    opt_state = client_optimizer.init(server_params)
    client_step_state = {
        'params': server_params,
        'opt_state': opt_state,
        'rng': client_rng,
        'server_params': server_params,
    }
    return client_step_state

  def client_step(client_step_state, batch):
    rng, use_rng = jax.random.split(client_step_state['rng'])
    grads = grad_fn(client_step_state['params'],
                    client_step_state['server_params'], batch, use_rng)
    opt_state, params = client_optimizer.apply(grads,
                                               client_step_state['opt_state'],
                                               client_step_state['params'])
    next_client_step_state = {
        'params': params,
        'opt_state': opt_state,
        'rng': rng,
        'server_params': client_step_state['server_params'],
    }
    return next_client_step_state

  def client_final(server_params, client_step_state):
    delta_params = jax.tree_util.tree_map(lambda a, b: a - b,
                                               server_params,
                                               client_step_state['params'])
    return delta_params

  return for_each_client.for_each_client(client_init, client_step, client_final)


def fed_prox(per_example_loss: Callable[[Params, BatchExample, PRNGKey],
                                        jnp.ndarray],
             client_optimizer: optimizers.Optimizer,
             server_optimizer: optimizers.Optimizer,
             client_batch_hparams: client_datasets.ShuffleRepeatBatchHParams,
             proximal_weight: float) -> federated_algorithm.FederatedAlgorithm:
  """Builds FedProx.

  Args:
    per_example_loss: A function from (params, batch, rng) to a vector of per
      example loss values. This will be combined with a proximal term based on
      server params weighted by proximal_weight.
    client_optimizer: Optimizer for local client training.
    server_optimizer: Optimizer for server update.
    client_batch_hparams: Hyperparameters for batching client dataset for train.
    proximal_weight: Weight for proximal term. 0 weight is FedAvg.

  Returns:
    FederatedAlgorithm
  """

  def fed_prox_loss(params, server_params, batch, rng):
    example_loss = per_example_loss(params, batch, rng)
    proximal_loss = 0.5 * proximal_weight * tree_util.tree_l2_squared(
        jax.tree_util.tree_map(lambda a, b: a - b, server_params, params))
    return jnp.mean(example_loss + proximal_loss)

  grad_fn = jax.grad(fed_prox_loss)
  train_for_each_client = create_train_for_each_client(grad_fn,
                                                       client_optimizer)

  def init(params: Params) -> ServerState:
    opt_state = server_optimizer.init(params)
    return ServerState(params, opt_state)

  def apply(
      server_state: ServerState,
      clients: Sequence[Tuple[federated_data.ClientId,
                              client_datasets.ClientDataset, PRNGKey]]
  ) -> Tuple[ServerState, Mapping[federated_data.ClientId, Any]]:
    client_num_examples = {cid: len(cds) for cid, cds, _ in clients}
    batch_clients = [(cid, cds.shuffle_repeat_batch(client_batch_hparams), crng)
                     for cid, cds, crng in clients]
    client_diagnostics = {}
    # Running weighted mean of client updates. We do this iteratively to avoid
    # loading all the client outputs into memory since they can be prohibitively
    # large depending on the model parameters size.
    delta_params_sum = tree_util.tree_zeros_like(server_state.params)
    num_examples_sum = 0.
    for client_id, delta_params in train_for_each_client(
        server_state.params, batch_clients):
      num_examples = client_num_examples[client_id]
      delta_params_sum = tree_util.tree_add(
          delta_params_sum, tree_util.tree_weight(delta_params, num_examples))
      num_examples_sum += num_examples
      # We record the l2 norm of client updates as an example, but it is not
      # required for the algorithm.
      client_diagnostics[client_id] = {
          'delta_l2_norm': tree_util.tree_l2_norm(delta_params)
      }
    mean_delta_params = tree_util.tree_inverse_weight(delta_params_sum,
                                                      num_examples_sum)
    server_state = server_update(server_state, mean_delta_params)
    return server_state, client_diagnostics

  def server_update(server_state, mean_delta_params):
    opt_state, params = server_optimizer.apply(mean_delta_params,
                                               server_state.opt_state,
                                               server_state.params)
    return ServerState(params, opt_state)

  return federated_algorithm.FederatedAlgorithm(init, apply)
