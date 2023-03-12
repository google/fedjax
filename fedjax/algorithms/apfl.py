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
"""Adaptive personalized federated learning implementation using fedjax.core.

  Adaptive Personalized Federated Learning
    Yuyang Deng, Mohammad Mahdi Kamani, Mehrdad Mahdavi. CoRR 2020.
    https://arxiv.org/abs/2003.13461
"""

from functools import partial
from typing import Any, Callable, Mapping, Sequence, Tuple

from fedjax.core import client_datasets
from fedjax.core import dataclasses
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


@dataclasses.dataclass
class ClientState:
  """State of client maintained over rounds.
  Attributes:
    params: A pytree representing the client model parameters.
    interpolation_coefficients: A pytree representing the client interpolation coefficient.
  """
  params: Params
  interpolation_coefficients: Params


@dataclasses.dataclass
class ServerState:
  """State of server passed between rounds.
  Attributes:
    params: A pytree representing the server model parameters.
    opt_state: A pytree representing the server optimizer state.
    client_states: A dict of pytrees representing client states.
  """
  params: Params
  opt_state: optimizers.OptState
  client_states: Mapping[federated_data.ClientId, ClientState]


@jax.jit
def interpolate_params(interpolation_coefficients, client_params, server_params):
  """Calculates the laywerwise interpolation of the client and server parameters."""

  return jax.tree_util.tree_map(
    lambda a, b, c: a * b + (1 - a) * c, interpolation_coefficients, client_params, server_params
  )


@jax.jit
def interpolation_grad_fn(grads, client_params, server_params):
  """Calculates the interpolation gradient using the client and server parameters."""

  def subtract(x, y):
    return jax.tree_util.tree_map(jnp.subtract, x, y)

  def inner(x, y):
    return jax.tree_util.tree_map(lambda a, b: jnp.tensordot(a, b, axes=len(a.shape)), x, y)

  return inner(subtract(client_params, server_params), grads)


def create_train_for_each_client(grad_fn, client_optimizer):
  """Builds client_init, client_step, client_final for for_each_client."""

  def client_init(server_params, client_input):
    return {
      'server_params': server_params,
      'server_opt_state': client_optimizer.init(server_params),
      'client_opt_state': client_optimizer.init(client_input['state'].params),
      'interpolation_opt_state': client_optimizer.init(client_input['state'].interpolation_coefficients),
      'rng': client_input['rng'],
      'state': client_input['state'],
    }

  def client_step(client_step_state, batch):
    rng, server_rng, client_rng = jax.random.split(client_step_state['rng'], 3)

    personalized_params = interpolate_params(
      client_step_state['state'].interpolation_coefficients,
      client_step_state['state'].params,
      client_step_state['server_params'])

    server_grads = grad_fn(client_step_state['server_params'], batch, server_rng)
    client_grads = grad_fn(personalized_params, batch, client_rng)
    interpolation_grads = interpolation_grad_fn(
      client_grads,
      client_step_state['state'].params,
      client_step_state['server_params'])

    server_opt_state, server_params = client_optimizer.apply(
      server_grads,
      client_step_state['server_opt_state'],
      client_step_state['server_params'])
    client_opt_state, client_params = client_optimizer.apply(
      client_grads,
      client_step_state['client_opt_state'],
      client_step_state['state'].params)
    interpolation_opt_state, interpolation_coefficients = client_optimizer.apply(
      interpolation_grads,
      client_step_state['interpolation_opt_state'],
      client_step_state['state'].interpolation_coefficients)

    interpolation_coefficients = jax.tree_util.tree_map(
      partial(jnp.clip, a_min=0, a_max=1),
      interpolation_coefficients)

    return {
      'server_params': server_params,
      'server_opt_state': server_opt_state,
      'client_opt_state': client_opt_state,
      'interpolation_opt_state': interpolation_opt_state,
      'rng': rng,
      'state': ClientState(
        params=client_params,
        interpolation_coefficients=interpolation_coefficients
      ),
    }

  def client_final(server_params, client_step_state):
    return {
      'state': client_step_state['state'],
      'delta_params': jax.tree_util.tree_map(
        jnp.subtract,
        server_params,
        client_step_state['server_params']),
    }

  return for_each_client.for_each_client(client_init, client_step, client_final)


def adaptive_personalized_federated_learning(
    grad_fn: Callable[[Params, BatchExample, PRNGKey], Grads],
    client_optimizer: optimizers.Optimizer,
    server_optimizer: optimizers.Optimizer,
    client_batch_hparams: client_datasets.ShuffleRepeatBatchHParams,
    client_coefficient: float
) -> federated_algorithm.FederatedAlgorithm:
  """Builds adaptive personalized federated learning.

  Args:
    grad_fn: A function from (params, batch_example, rng) to gradients.
      This can be created with :func:`fedjax.core.model.model_grad`.
    client_optimizer: Optimizer for local client training.
    server_optimizer: Optimizer for server update.
    client_batch_hparams: Hyperparameters for batching client dataset for train.
    client_coefficient: Initial interpolation coefficient for local client training.

  Returns:
    FederatedAlgorithm
  """
  train_for_each_client = create_train_for_each_client(grad_fn, client_optimizer)

  def init(params: Params) -> ServerState:
    return ServerState(
      params=params,
      opt_state=server_optimizer.init(params),
      client_states={})

  def apply(
      server_state: ServerState,
      clients: Sequence[Tuple[federated_data.ClientId, client_datasets.ClientDataset, PRNGKey]]
  ) -> Tuple[ServerState, Mapping[federated_data.ClientId, Any]]:
    client_num_examples = {cid: len(cds) for cid, cds, _ in clients}
    client_default_state = ClientState(
      params=server_state.params,
      interpolation_coefficients=jax.tree_util.tree_map(
        lambda _: client_coefficient,
        server_state.params))

    batch_clients = (
      (cid,
       cds.shuffle_repeat_batch(client_batch_hparams),
       {'rng': crng, 'state': server_state.client_states.get(cid, client_default_state)})
      for cid, cds, crng in clients
    )

    client_diagnostics = {}
    # Running weighted mean of client updates. We do this iteratively to avoid
    # loading all the client outputs into memory since they can be prohibitively
    # large depending on the model parameters size.
    delta_params_sum = tree_util.tree_zeros_like(server_state.params)
    num_examples_sum = 0.
    for client_id, client_output in train_for_each_client(server_state.params, batch_clients):
      delta_params = client_output['delta_params']
      server_state.client_states[client_id] = client_output['state']
      num_examples = client_num_examples[client_id]
      delta_params_sum = tree_util.tree_add(delta_params_sum, tree_util.tree_weight(delta_params, num_examples))
      num_examples_sum += num_examples
      # We record the l2 norm of client updates as an example, but it is not
      # required for the algorithm.
      client_diagnostics[client_id] = {
          'delta_l2_norm': tree_util.tree_l2_norm(delta_params)
      }
    mean_delta_params = tree_util.tree_inverse_weight(delta_params_sum,num_examples_sum)
    server_state = server_update(server_state, mean_delta_params)

    return server_state, client_diagnostics

  def server_update(server_state, mean_delta_params):
    opt_state, params = server_optimizer.apply(
      mean_delta_params,
      server_state.opt_state,
      server_state.params)

    return ServerState(params, opt_state, server_state.client_states)

  return federated_algorithm.FederatedAlgorithm(init, apply)


def create_eval_for_each_client(model: models.Model):
  """Builds client_init, client_step, client_final for for_each_client."""

  def client_init(server_params, client_input):
    params = interpolate_params(
      client_input.interpolation_coefficients,
      client_input.params,
      server_params)

    return {
      'params': params,
      'stat': {k: metric.zero() for k, metric in model.eval_metrics.items()}
    }

  def client_step(client_step_state, batch):
    stat = models._evaluate_model_step(
      model,
      client_step_state['params'],
      batch,
      client_step_state['stat'])

    return {
      'params': client_step_state['params'],
      'stat': stat
    }

  def client_final(server_params, client_step_state):
    del server_params
    return client_step_state['stat']

  return for_each_client.for_each_client(client_init, client_step, client_final)


def eval_adaptive_personalized_federated_learning(
    model: models.Model,
    client_batch_hparams: client_datasets.PaddedBatchHParams,
) -> Callable[
  [ServerState, Sequence[Tuple[federated_data.ClientId, client_datasets.ClientDataset]]],
  Sequence[Tuple[federated_data.ClientId, Mapping[str, jnp.array]]]
]:
  """Evaluates adaptive personalized federated learning.

  Args:
    model: Model
    client_batch_hparams: Hyperparameters for batching client dataset for eval.

  Returns:
    Callable
  """
  eval_for_each_client = create_eval_for_each_client(model)

  def __fn(
      server_state: ServerState,
      clients: Sequence[Tuple[federated_data.ClientId, client_datasets.ClientDataset]]
  ) -> Sequence[Tuple[federated_data.ClientId, Mapping[str, jnp.array]]]:
    client_default_state = ClientState(
      params=server_state.params,
      interpolation_coefficients=tree_util.tree_zeros_like(server_state.params))

    batch_clients = (
      (cid,
       cds.padded_batch(client_batch_hparams),
       server_state.client_states.get(cid, client_default_state))
      for cid, cds in clients
    )

    return eval_for_each_client(server_state.params, batch_clients)

  return __fn
