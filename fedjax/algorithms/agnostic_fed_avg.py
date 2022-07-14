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
"""AgnosticFedAvg implementation.

Communication-Efficient Agnostic Federated Averaging
    Jae Ro, Mingqing Chen, Rajiv Mathews, Mehryar Mohri, Ananda Theertha Suresh
    https://arxiv.org/abs/2104.02748
"""

from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple

from fedjax.core import client_datasets
from fedjax.core import dataclasses
from fedjax.core import federated_algorithm
from fedjax.core import federated_data
from fedjax.core import for_each_client
from fedjax.core import optimizers
from fedjax.core import tree_util
from fedjax.core.typing import BatchExample
from fedjax.core.typing import Params
from fedjax.core.typing import PRNGKey
from fedjax.core import util

import jax
import jax.numpy as jnp


def create_domain_metrics_for_each_client(
    per_example_loss: Callable[[Params, BatchExample, PRNGKey], jnp.ndarray],
    num_domains: int,
    regularizer: Optional[Callable[[Params], jnp.ndarray]] = None):
  """Creates for_each_client function for domain metrics calculation."""

  def client_init(shared_input, client_rng):
    step_state = {
        'params': shared_input['params'],
        'rng': client_rng,
        'domain_loss': jnp.zeros(num_domains),
        'domain_num': jnp.zeros(num_domains),
    }
    return step_state

  def client_step(step_state, batch):
    rng, use_rng = jax.random.split(step_state['rng'])
    example_mask = batch[client_datasets.EXAMPLE_MASK_KEY]
    example_loss = (
        per_example_loss(step_state['params'], batch, use_rng) * example_mask)
    domain_loss = jax.ops.segment_sum(example_loss, batch['domain_id'],
                                      num_domains)
    if regularizer is not None:
      domain_loss += regularizer(step_state['params'])
    domain_num = jax.ops.segment_sum(
        example_mask.astype(jnp.float32), batch['domain_id'], num_domains)
    next_step_state = {
        'params': step_state['params'],
        'rng': rng,
        'domain_loss': step_state['domain_loss'] + domain_loss,
        'domain_num': step_state['domain_num'] + domain_num
    }
    return next_step_state

  def client_final(shared_input, step_state):
    client_output = {
        'domain_loss': step_state['domain_loss'],
        'domain_num': step_state['domain_num'],
        'beta': jnp.sum(shared_input['alpha'] * step_state['domain_num'])
    }
    return client_output

  return for_each_client.for_each_client(client_init, client_step, client_final)


def create_scaled_loss(
    per_example_loss: Callable[[Params, BatchExample, PRNGKey], jnp.ndarray],
    num_domains: int,
    regularizer: Optional[Callable[[Params], jnp.ndarray]] = None):
  """Creates domain scaled loss function for agnostic client training."""

  def scaled_loss(params, batch, rng, alpha, beta):
    did = batch['domain_id']
    # Training batches produced by `ClientDataset.shuffle_repeat_batch` are not
    # padded so there's no need to worry about example masking.
    example_loss = per_example_loss(params, batch, rng)
    # \sum_{j \in b \cap \h{\sD}_i} \L(w, x_j, y_j)
    domain_sum_loss = jax.ops.segment_sum(example_loss, did, num_domains)
    loss = jnp.sum(alpha * domain_sum_loss) / beta
    if regularizer is not None:
      loss += regularizer(params)
    return loss

  return scaled_loss


def create_train_for_each_client(
    per_example_loss: Callable[[Params, BatchExample, PRNGKey], jnp.ndarray],
    client_optimizer: optimizers.Optimizer,
    num_domains: int,
    regularizer: Optional[Callable[[Params], jnp.ndarray]] = None):
  """Creates for_each_client for client training."""
  grad_fn = jax.grad(
      create_scaled_loss(per_example_loss, num_domains, regularizer))

  def client_init(shared_input, client_input):
    step_state = {
        'params': shared_input['params'],
        'alpha': shared_input['alpha'],
        'opt_state': client_optimizer.init(shared_input['params']),
        'rng': client_input['rng'],
        'beta': client_input['beta']
    }
    return step_state

  def client_step(step_state, batch):
    rng, use_rng = jax.random.split(step_state['rng'])
    grads = grad_fn(step_state['params'], batch, use_rng, step_state['alpha'],
                    step_state['beta'])
    opt_state, params = client_optimizer.apply(grads, step_state['opt_state'],
                                               step_state['params'])
    next_step_state = {
        **step_state,
        'params': params,
        'opt_state': opt_state,
        'rng': rng
    }
    return next_step_state

  def client_final(shared_input, step_state):
    delta_params = jax.tree_util.tree_map(lambda a, b: a - b,
                                               shared_input['params'],
                                               step_state['params'])
    return delta_params

  return for_each_client.for_each_client(client_init, client_step, client_final)


def update_domain_weights(domain_weights: jnp.ndarray, domain_loss: jnp.ndarray,
                          domain_learning_rate: float,
                          domain_algorithm: str) -> jnp.ndarray:
  """Updates domain weights following input algorithm."""
  if domain_algorithm == 'eg':
    new_domain_weights = domain_weights * jnp.exp(
        domain_learning_rate * domain_loss)
    new_domain_weights = jnp.maximum(new_domain_weights,
                                     jnp.zeros_like(new_domain_weights))
    return new_domain_weights / jnp.sum(new_domain_weights)
  elif domain_algorithm == 'none':
    return domain_weights
  else:
    raise ValueError(f'Unsupported domain algorithm {domain_algorithm!r}.')


@dataclasses.dataclass
class ServerState:
  """State of server for AgnosticFedAvg passed between rounds.

  Attributes:
    params: A pytree representing the server model parameters.
    opt_state: A pytree representing the server optimizer state.
    domain_weights: Weights per domain applied to weight in weighted average.
    domain_window: Sliding window keeping track of domain number of examples for
      the last window size rounds.
  """
  params: Params
  opt_state: optimizers.OptState
  domain_weights: jnp.ndarray
  domain_window: List[jnp.ndarray]


def agnostic_federated_averaging(
    per_example_loss: Callable[[Params, BatchExample, PRNGKey], jnp.ndarray],
    client_optimizer: optimizers.Optimizer,
    server_optimizer: optimizers.Optimizer,
    client_batch_hparams: client_datasets.ShuffleRepeatBatchHParams,
    domain_batch_hparams: client_datasets.PaddedBatchHParams,
    init_domain_weights: Sequence[float],
    domain_learning_rate: float,
    domain_algorithm: str = 'eg',
    domain_window_size: int = 1,
    init_domain_window: Optional[Sequence[float]] = None,
    regularizer: Optional[Callable[[Params], jnp.ndarray]] = None
) -> federated_algorithm.FederatedAlgorithm:
  """Builds agnostic federated averaging.

  Agnostic federated averaging requires input
  :class:`fedjax.core.client_datasets.ClientDataset` examples to contain
  a feature named "domain_id", which stores the integer domain id in
  [0, num_domains).
  For example, for Stack Overflow, each example post can be either a question or
  an answer, so there are two possible domain ids (question = 0; answer = 1).

  Args:
    per_example_loss: A function from (params, batch_example, rng) to a vector
      of loss values for each example in the batch. This is used in both the
      domain metrics computation and gradient descent training.
    client_optimizer: Optimizer for local client training.
    server_optimizer: Optimizer for server update.
    client_batch_hparams: Hyperparameters for client dataset for training.
    domain_batch_hparams: Hyperparameters for client dataset domain metrics
      calculation.
    init_domain_weights: Initial weights per domain that must sum to 1.
    domain_learning_rate: Learning rate for domain weight update.
    domain_algorithm: Algorithm used to update domain weights each round. One of
      'eg', 'none'.
    domain_window_size: Size of sliding window keeping track of number of
      examples per domain over multiple rounds.
    init_domain_window: Initial values for domain window. Defaults to ones.
    regularizer: Optional regularizer that only depends on params.

  Returns:
    FederatedAlgorithm.

  Raises:
    ValueError: If ``init_domain_weights`` does not sum to 1 or if
      ``init_domain_weights`` and ``init_domain_window`` are unequal lengths.
  """
  if abs(sum(init_domain_weights) - 1) > 1e-6:
    raise ValueError('init_domain_weights must sum to approximately 1.')

  if init_domain_window is None:
    init_domain_window = jnp.ones_like(init_domain_weights)

  if len(init_domain_weights) != len(init_domain_window):
    raise ValueError(
        f'init_domain_weights and init_domain_window must be equal lengths.'
        f' {len(init_domain_weights)} != {len(init_domain_window)}'
    )

  num_domains = len(init_domain_weights)
  domain_metrics_for_each_client = create_domain_metrics_for_each_client(
      per_example_loss, num_domains)
  train_for_each_client = create_train_for_each_client(per_example_loss,
                                                       client_optimizer,
                                                       num_domains, regularizer)

  def init(params: Params) -> ServerState:
    opt_state = server_optimizer.init(params)
    domain_weights = jnp.array(init_domain_weights)
    domain_window = [jnp.array(init_domain_window)] * domain_window_size
    return ServerState(params, opt_state, domain_weights, domain_window)

  def apply(
      server_state: ServerState,
      clients: Sequence[Tuple[federated_data.ClientId,
                              client_datasets.ClientDataset, PRNGKey]]
  ) -> Tuple[ServerState, Mapping[federated_data.ClientId, Any]]:
    # α
    alpha = server_state.domain_weights / jnp.mean(
        jnp.asarray(server_state.domain_window), axis=0)
    # First pass to calculate initial domain loss, domain num, and scaling
    # weight β for each client. This doesn't involve any aggregation at the
    # server, so this step and training can be a single round of communication.
    domain_batch_clients = [(cid, cds.padded_batch(domain_batch_hparams), crng)
                            for cid, cds, crng in clients]
    shared_input = {'params': server_state.params, 'alpha': alpha}
    # L^k, N^k, β^k
    client_domain_metrics = dict(
        domain_metrics_for_each_client(shared_input, domain_batch_clients))
    # Train for each client using scaling weights α and β.
    batch_clients = []
    for cid, cds, crng in clients:
      client_input = {'rng': crng, 'beta': client_domain_metrics[cid]['beta']}
      batch_clients.append(
          (cid, cds.shuffle_repeat_batch(client_batch_hparams), client_input))

    client_diagnostics = {}
    # Mean delta params across clients.
    delta_params_sum = tree_util.tree_zeros_like(server_state.params)
    weight_sum = 0.
    # w^k
    for cid, delta_params in train_for_each_client(shared_input, batch_clients):
      weight = client_domain_metrics[cid]['beta']
      delta_params_sum = tree_util.tree_add(
          delta_params_sum, tree_util.tree_weight(delta_params, weight))
      weight_sum += weight
      client_diagnostics[cid] = {
          'delta_l2_norm': tree_util.tree_l2_norm(delta_params)
      }
    mean_delta_params = tree_util.tree_inverse_weight(delta_params_sum,
                                                      weight_sum)
    # Sum domain metrics across clients.
    sum_domain_loss = tree_util.tree_sum(
        d['domain_loss'] for d in client_domain_metrics.values())
    sum_domain_num = tree_util.tree_sum(
        d['domain_num'] for d in client_domain_metrics.values())
    server_state = server_update(server_state, mean_delta_params,
                                 sum_domain_loss, sum_domain_num)
    return server_state, client_diagnostics

  def server_update(server_state, mean_delta_params, sum_domain_loss,
                    sum_domain_num):
    opt_state, params = server_optimizer.apply(mean_delta_params,
                                               server_state.opt_state,
                                               server_state.params)
    mean_domain_loss = util.safe_div(sum_domain_loss, sum_domain_num)
    domain_weights = update_domain_weights(server_state.domain_weights,
                                           mean_domain_loss,
                                           domain_learning_rate,
                                           domain_algorithm)
    domain_window = server_state.domain_window[1:] + [sum_domain_num]
    return ServerState(params, opt_state, domain_weights, domain_window)

  return federated_algorithm.FederatedAlgorithm(init, apply)
