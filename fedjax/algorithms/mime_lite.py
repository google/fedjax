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
"""Mime Lite implementation.

Mime: Mimicking Centralized Stochastic Algorithms in Federated Learning
    Sai Praneeth Karimireddy, Martin Jaggi, Satyen Kale, Mehryar Mohri,
    Sashank J. Reddi, Sebastian U. Stich, Ananda Theertha Suresh
    https://arxiv.org/abs/2008.03606

Reuses :class:`fedjax.algorithms.mime.ServerState`
"""

from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

from fedjax.algorithms import mime

from fedjax.core import client_datasets
from fedjax.core import federated_algorithm
from fedjax.core import federated_data
from fedjax.core import for_each_client
from fedjax.core import models
from fedjax.core import optimizers
from fedjax.core import tree_util
from fedjax.core.typing import BatchExample
from fedjax.core.typing import Params
from fedjax.core.typing import PRNGKey

import jax
import jax.numpy as jnp

Grads = Params


def create_train_for_each_client(
    grad_fn: Callable[[Params, BatchExample, PRNGKey], Grads],
    base_optimizer: optimizers.Optimizer):
  """Builds for_each_client for client training."""

  def client_init(shared_input, client_rng):
    step_state = {
        'params': shared_input['params'],
        'opt_state': shared_input['opt_state'],
        'rng': client_rng
    }
    return step_state

  def client_step(step_state, batch):
    rng, use_rng = jax.random.split(step_state['rng'])
    grads = grad_fn(step_state['params'], batch, use_rng)
    _, params = base_optimizer.apply(grads, step_state['opt_state'],
                                     step_state['params'])
    next_step_state = {
        'params': params,
        'opt_state': step_state['opt_state'],
        'rng': rng
    }
    return next_step_state

  def client_final(shared_input, step_state):
    delta_params = jax.tree_util.tree_multimap(lambda a, b: a - b,
                                               shared_input['params'],
                                               step_state['params'])
    return delta_params

  return for_each_client.for_each_client(client_init, client_step, client_final)


def mime_lite(
    per_example_loss: Callable[[Params, BatchExample, PRNGKey], jnp.ndarray],
    base_optimizer: optimizers.Optimizer,
    client_batch_hparams: client_datasets.ShuffleRepeatBatchHParams,
    grads_batch_hparams: client_datasets.PaddedBatchHParams,
    server_learning_rate: float,
    regularizer: Optional[Callable[[Params], jnp.ndarray]] = None,
    client_delta_clip_norm: Optional[float] = None,
) -> federated_algorithm.FederatedAlgorithm:
  """Builds mime lite.

  Args:
    per_example_loss: A function from (params, batch_example, rng) to a vector
      of loss values for each example in the batch. This is used in both the
      server gradient computation and gradient descent training.
    base_optimizer: Base optimizer to mimic.
    client_batch_hparams: Hyperparameters for batching client dataset for train.
    grads_batch_hparams: Hyperparameters for batching client dataset for server
      gradient computation.
    server_learning_rate: Server learning rate.
    regularizer: Optional regularizer that only depends on params.
    client_delta_clip_norm: Maximum allowed global norm per client update.
      Defaults to no clipping.

  Returns:
    FederatedAlgorithm
  """
  grad_fn = models.grad(per_example_loss, regularizer)
  grads_for_each_client = mime.create_grads_for_each_client(grad_fn)
  train_for_each_client = create_train_for_each_client(grad_fn, base_optimizer)

  def init(params: Params) -> mime.ServerState:
    opt_state = base_optimizer.init(params)
    return mime.ServerState(params, opt_state)

  def apply(
      server_state: mime.ServerState,
      clients: Sequence[Tuple[federated_data.ClientId,
                              client_datasets.ClientDataset, PRNGKey]]
  ) -> Tuple[mime.ServerState, Mapping[federated_data.ClientId, Any]]:
    # Training across clients using fixed optimizer state.
    client_diagnostics = {}
    client_num_examples = {cid: len(cds) for cid, cds, _ in clients}
    batch_clients = [(cid, cds.shuffle_repeat_batch(client_batch_hparams), crng)
                     for cid, cds, crng in clients]
    shared_input = {
        'params': server_state.params,
        'opt_state': server_state.opt_state
    }
    # Running weighted mean of client updates.
    delta_params_sum = tree_util.tree_zeros_like(server_state.params)
    num_examples_sum = 0.
    for client_id, delta_params in train_for_each_client(
        shared_input, batch_clients):
      num_examples = client_num_examples[client_id]
      client_diagnostics[client_id] = {
          'delta_l2_norm': tree_util.tree_l2_norm(delta_params)
      }
      if client_delta_clip_norm is not None:
        delta_params = tree_util.tree_clip_by_global_norm(
            delta_params, client_delta_clip_norm)
        client_diagnostics[client_id]['clipped_delta_l2_norm'] = (
            tree_util.tree_l2_norm(delta_params))
        client_diagnostics[client_id]['clipped'] = jnp.not_equal(
            client_diagnostics[client_id]['delta_l2_norm'],
            client_diagnostics[client_id]['clipped_delta_l2_norm'])
      delta_params_sum = tree_util.tree_add(
          delta_params_sum, tree_util.tree_weight(delta_params, num_examples))
      num_examples_sum += num_examples
    mean_delta_params = tree_util.tree_inverse_weight(delta_params_sum,
                                                      num_examples_sum)

    # Compute full-batch gradient at server params on train data.
    grads_batch_clients = [(cid, cds.padded_batch(grads_batch_hparams), crng)
                           for cid, cds, crng in clients]
    grads_sum_total, num_sum_total = tree_util.tree_sum(
        (co for _, co in grads_for_each_client(server_state.params,
                                               grads_batch_clients)))
    server_grads = tree_util.tree_inverse_weight(grads_sum_total, num_sum_total)

    server_state = server_update(server_state, server_grads, mean_delta_params)
    return server_state, client_diagnostics

  def server_update(server_state, server_grads, mean_delta_params):
    # Server params uses weighted average of client updates, scaled by the
    # server_learning_rate.
    params = jax.tree_util.tree_multimap(
        lambda p, q: p - server_learning_rate * q, server_state.params,
        mean_delta_params)
    opt_state, _ = base_optimizer.apply(server_grads, server_state.opt_state,
                                        server_state.params)
    return mime.ServerState(params, opt_state)

  return federated_algorithm.FederatedAlgorithm(init, apply)
