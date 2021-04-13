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
"""Transformations and utilities for specifying work for each client.

Many of these utilities work with and require PyTrees as input and output.
This is required when working with the underlying JAX transformations like
`jax.pmap` and `jax.jit`. For more information on what PyTrees are, refer to
https://jax.readthedocs.io/en/latest/pytrees.html.
"""

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from fedjax.experimental.typing import BatchExample

import jax

PyTree = Any

# Server state that is passed to the client init to initialize the client state.
# This will contain aggregate information like the global model parameters that
# are then used as the starting point for client state.
ServerState = PyTree
# Optional persistent client state that can be passed alongside server state to
# the client init. For stateful client federated algorithms (usually in the
# cross-silo setting), this is how per client information will be passed along.
PersistentClientState = PyTree
# Intermittent client state passed as input and output for each client step.
# This will typically contain model parameters and optimizer state that are
# updated at each client step.
ClientState = PyTree
# Step results can be used to record any metrics over the course of running the
# for_each_client loop. For example, it can be used to record train metrics for
# each client like train loss and gradient norm.
StepResult = PyTree
# Final output from the for_each_client loop. This is usually a subset of the
# client state. However, more meaningful transformations can be done like adding
# noise to the final output for differential privacy.
ClientOutput = PyTree

ClientInit = Union[Callable[[ServerState], ClientState],
                   Callable[[ServerState, PersistentClientState], ClientState]]
ClientStep = Callable[[ClientState, BatchExample], Tuple[ClientState,
                                                         StepResult]]
ClientFinal = Callable[[ClientState], ClientOutput]

ClientId = str
# TODO(wuke): Replace this with the actual client dataset type.
ClientDatas = Iterable[Tuple[ClientId, Any]]
ForEachClient = Callable[
    [ClientDatas, ServerState, Optional[Dict[ClientId, PersistentClientState]]],
    Iterable[Tuple[ClientId, ClientOutput, List[StepResult]]]]


def for_each_client_jit(client_init: ClientInit, client_step: ClientStep,
                        client_final: ClientFinal) -> ForEachClient:
  """Creates a for each client function backed by `jax.jit`."""
  client_init_jit = jax.jit(client_init)
  client_step_jit = jax.jit(client_step)
  client_final_jit = jax.jit(client_final)

  def run(client_datas, server_state, persistent_client_states=None):
    for client_id, client_data in client_datas:
      step_results = []
      # pytype: disable=wrong-arg-count
      if persistent_client_states is None:
        client_state = client_init_jit(server_state)
      else:
        persistent_client_state = persistent_client_states[client_id]
        client_state = client_init_jit(server_state, persistent_client_state)
      # pytype: enable=wrong-arg-count
      for batch in client_data:
        client_state, step_result = client_step_jit(client_state, batch)
        step_results.append(step_result)
      client_output = client_final_jit(client_state)
      yield client_id, client_output, step_results

  return run


def for_each_client(client_init: ClientInit,
                    client_step: ClientStep,
                    client_final: ClientFinal = lambda s: s) -> ForEachClient:
  """Creates a function which maps over client datasets.

  For example, `for_each_client` could be used to define how to run client
  updates for each client in a federated training round. Another common use case
  of `for_each_client` is to run evaluation per client for a given set of model
  parameters.

  The underlying backend for `for_each_client` can differ depending on the
  available devices. For example, if multiple devies are available (e.g. TPU),
  `for_each_client` will use `jax.pmap` to parallelize across devices. It's also
  possible to manually specify which backend to use (helpful for debugging).

  The expected usage of `for_each_client` is as follows:

  ```python
  # Map over clients and count how many points are greater than `limit` for
  # each client. In addition to the total `count`, we'll also keep track of the
  # `num` per step in our step results.

  def client_init(server_state):
    return {'limit': server_state['limit'], 'count': 0.}

  def client_step(client_state, batch):
    limit = client_state['limit']
    num = jnp.sum(batch['x'] > limit)
    client_state = {'limit': limit, 'count': client_state['count'] + num}
    step_result = {'num': num}
    return client_state, step_result

  def client_final(client_state):
    return client_state['count']

  # Three clients with different data (`client_datasets`)
  # and starting counts (`client_infos`).
  client_datasets = [
      ('cid0', [{'x': jnp.array([1, 2, 3, 4])}, {'x': jnp.array([1, 2, 3])}]),
      ('cid1', [{'x': jnp.array([1, 2])}, {'x': jnp.array([1, 2, 3, 4, 5])}]),
      ('cid2', [{'x': jnp.array([1])}]),
  ]
  server_state = {'limit': jnp.array(2)}

  func = fedjax.experimental.for_each_client.for_each_client(
      client_init, client_step, client_final)
  print(list(func(client_datasets, server_state)))
  # [
  #   ('cid0', 3, [{'num': 2}, {'num': 1}]),
  #   ('cid1', 3, [{'num': 0}, {'num': 3}]),
  #   ('cid2', 0, [{'num': 0}]),
  # ]
  ```

  The same example with persistent client states:

  ```python
  def client_init_with_persistent_state(server_state, persistent_client_state):
    return {
        'limit': server_state['limit'],
        'count': persistent_client_state['count'],
    }

  persistent_client_states = {
      'cid0': {'count': jnp.array(2)},
      'cid1': {'count': jnp.array(0)},
      'cid2': {'count': jnp.array(1)},
  }

  func = fedjax.experimental.for_each_client.for_each_client(
      client_init_with_persistent_state, client_step, client_final)
  print(list(func(client_datasets, server_state, persistent_client_states)))
  # [
  #   ('cid0', 5, [{'num': 2}, {'num': 1}]),
  #   ('cid1', 3, [{'num': 0}, {'num': 3}]),
  #   ('cid2', 1, [{'num': 0}]),
  # ]
  ```

  Args:
    client_init: Function that initializes the internal client state from the
      starting server state as well as an optional persistent client state.
      The server state contains aggregate information like the global model
      parameters that are then used as the starting point to initialize the
      client state.
      The optional persistent client state is per client information that is
      expected to be persisted over multiple calls by the caller.
      The initialized internal client state is fed as input and output from
      `client_step` and `client_final`. Client state usually contains things
      like the model parameters and optimizer state that are updated at each
      `client_step`.
      This will be run once for each client.
    client_step: Function that takes the internal client state and a batch of
      examples as input and outputs a (possibly updated) client state along with
      any per step results. Per step results are usually diagnostics like train
      loss or gradient norm.
      This will be run for each batch for each client.
    client_final: Function that applies the final transformation on the internal
      client state to the desired, final client output. More meaningful
      transformations can be done here, like model update clipping.
      Defaults to just returning client state.
      This will be run once for each client.

  Returns:
    A for each client function that takes the client datas to map over, server
      state, and optional persistent client states as input and returns the
      outputs per client as specified in `client_final` along with any
      per client per step results.
  """
  for_each_client_backend = for_each_client_jit
  return for_each_client_backend(client_init, client_step, client_final)
