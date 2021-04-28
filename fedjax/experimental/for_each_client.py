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

from typing import Callable, Iterable, List, Tuple, Union

from fedjax.experimental import federated_data
from fedjax.experimental.typing import BatchExample
from fedjax.experimental.typing import PyTree

import jax

# Shared input that is passed to the client init that is shared across all
# clients. For example, this could be the shared global model parameters that
# are then used as the starting point for per client training.
SharedInput = PyTree
# Client specific input passed into client init that is different for each
# client. For example, this could be the starting PRNG seed for training a given
# client. Additionally, for stateful client federated algorithms (usually in the
# cross-silo setting), this is how per client information will be passed along.
ClientInput = PyTree

# Intermittent client state passed as input and output for each client step.
# This will typically contain model parameters and optimizer state that are
# updated at each client step.
ClientStepState = PyTree
# Step results can be used to record any metrics over the course of running the
# for_each_client loop. For example, it can be used to record train metrics for
# each client like gradient norm.
ClientStepResult = PyTree

# Final output from the for_each_client loop. This is usually a subset of the
# client step state. However, more meaningful transformations can be done like
# model update clipping.
ClientOutput = PyTree

ClientInit = Callable[[SharedInput, ClientInput], ClientStepState]
ClientStep = Callable[[ClientStepState, BatchExample], Tuple[ClientStepState,
                                                             ClientStepResult]]
ClientFinal = Callable[[SharedInput, ClientStepState], ClientOutput]

ForEachClient = Callable[[
    Iterable[Tuple[federated_data.ClientId, Iterable[BatchExample],
                   ClientInput]], SharedInput
], Iterable[Tuple[federated_data.ClientId, ClientOutput,
                  List[ClientStepResult]]]]


def for_each_client_jit(client_init: ClientInit,
                        client_step: ClientStep,
                        client_final: ClientFinal) -> ForEachClient:
  """Creates a for each client function backed by `jax.jit`."""
  client_init_jit = jax.jit(client_init)
  client_step_jit = jax.jit(client_step)
  client_final_jit = jax.jit(client_final)

  def run(shared_input, clients):
    for client_id, client_batches, client_input in clients:
      client_step_results = []
      client_step_state = client_init_jit(shared_input, client_input)
      for batch in client_batches:
        client_step_state, client_step_result = client_step_jit(
            client_step_state, batch)
        client_step_results.append(client_step_result)
      client_output = client_final_jit(shared_input, client_step_state)
      yield client_id, client_output, client_step_results

  return run


# We leave the return type unannotated because there's no easy way to properly
# annotate it when it depends on the input value of with_step_result.
def for_each_client(client_init: ClientInit,
                    client_step: Union[ClientStep,
                                       Callable[[ClientStepState, BatchExample],
                                                ClientStepState]],
                    client_final: ClientFinal = lambda _, s: s,
                    with_step_result: bool = False):
  """Creates a function which maps over clients.

  For example, `for_each_client` could be used to define how to run client
  updates for each client in a federated training round. Another common use case
  of `for_each_client` is to run evaluation per client for a given set of model
  parameters.

  The underlying backend for `for_each_client` can differ depending on the
  available devices. For example, if multiple devies are available (e.g. TPU),
  `for_each_client` will use `jax.pmap` to parallelize across devices. It's also
  possible to manually specify which backend to use (helpful for debugging).

  The expected usage of `for_each_client` is as follows:

  ```
  # Map over clients and count how many points are greater than `limit` for
  # each client. Each client also has a different `start` that is specified via
  # client input.

  def client_init(shared_input, client_input):
    client_step_state = {
        'limit': shared_input['limit'],
        'count': client_input['start']
    }
    return client_step_state

  def client_step(client_step_state, batch):
    num = jnp.sum(batch['x'] > client_step_state['limit'])
    client_step_state = {
        'limit': client_step_state['limit'],
        'count': client_step_state['count'] + num
    }
    return client_step_state

  def client_final(shared_input, client_step_state):
    del shared_input  # Unused.
    return client_step_state['count']

  # Three clients with different data and starting counts.
  # clients = [(client_id, client_batches, client_input)]
  clients = [
      (b'cid0',
      [{'x': jnp.array([1, 2, 3, 4])}, {'x': jnp.array([1, 2, 3])}],
      {'start': jnp.array(2)}),
      (b'cid1',
      [{'x': jnp.array([1, 2])}, {'x': jnp.array([1, 2, 3, 4, 5])}],
      {'start': jnp.array(0)}),
      (b'cid2',
      [{'x': jnp.array([1])}],
      {'start': jnp.array(1)}),
  ]
  shared_input = {'limit': jnp.array(2)}

  func = fedjax.experimental.for_each_client.for_each_client(
      client_init, client_step, client_final)
  print(list(func(shared_input, clients)))
  # [(b'cid0', 5), (b'cid1', 3), (b'cid2', 1)]
  ```

  Here's the same example with per step results.

  ```
  # We'll also keep track of the `num` per step in our step results.

  def client_step_with_result(client_step_state, batch):
    num = jnp.sum(batch['x'] > client_step_state['limit'])
    client_step_state = {
        'limit': client_step_state['limit'],
        'count': client_step_state['count'] + num
    }
    client_step_result = {'num': num}
    return client_step_state, client_step_result

  func = fedjax.experimental.for_each_client.for_each_client(
      client_init, client_step_with_result, client_final, with_step_result=True)
  print(list(func(shared_input, clients)))
  # [
  #   (b'cid0', 5, [{'num': 2}, {'num': 1}]),
  #   (b'cid1', 3, [{'num': 0}, {'num': 3}]),
  #   (b'cid2', 1, [{'num': 0}]),
  # ]
  ```

  Args:
    client_init: Function that initializes the internal intermittent client step
      state from the share input and per client input. The shared input contains
      information like the global model parameters that are shared across all
      clients. The per client input is per client information. The initialized
      internal client step state is fed as intermittent input and output from
      `client_step` and `client_final`. This client step state usually contains
      the model parameters and optimizer state for each client that are updated
      at each `client_step`. This will be run once for each client.
    client_step: Function that takes the client step state and a batch of
      examples as input and outputs a (possibly updated) client step state.
      Optionally, per step results can also be returned as the second element if
      `with_step_result` is True. Per step results are usually diagnostics like
      gradient norm. This will be run for each batch for each client.
    client_final: Function that applies the final transformation on the internal
      client step state to the desired final client output. More meaningful
      transformations can be done here, like model update clipping. Defaults to
      just returning the client step state. This will be run once for each
      client.
    with_step_result: Indicates whether client_step returns a pair where the
      first element is considered the client output and the second element is
      the client step result.

  Returns:
    A for each client function that takes the per client inputs to map over and
      shared input and returns the outputs per client as specified in
      `client_final` along with optional per client per step results.
  """
  for_each_client_backend = for_each_client_jit
  if with_step_result:
    return for_each_client_backend(client_init, client_step, client_final)

  def client_step_with_result(client_step_state, batch):
    return client_step(client_step_state, batch), ()

  func = for_each_client_backend(client_init, client_step_with_result,
                                 client_final)

  def run(shared_input, clients):
    for client_id, client_output, _ in func(shared_input, clients):
      yield client_id, client_output

  return run
