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

import abc
import contextlib
import functools
import threading
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from fedjax.core import dataclasses
from fedjax.core.typing import BatchExample
from fedjax.core.typing import PyTree

import jax
import jax.numpy as jnp

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

# For the purpose of this module, any hashable type can be used as a client id.
ClientId = Any

ForEachClient = Callable[[
    Iterable[Tuple[ClientId, Iterable[BatchExample], ClientInput]], SharedInput
], Iterable[Tuple[ClientId, ClientOutput, List[ClientStepResult]]]]


class ForEachClientBackend(abc.ABC):

  @abc.abstractmethod
  def __call__(self, client_init: ClientInit, client_step: ClientStep,
               client_final: ClientFinal) -> ForEachClient:
    """Creates a `for_each_client` function."""


class ForEachClientJitBackend(ForEachClientBackend):
  """A straightforward for_each_client backend using jax.jit."""

  def __call__(self, client_init: ClientInit, client_step: ClientStep,
               client_final: ClientFinal) -> ForEachClient:
    """Creates a `for_each_client` function backed by `jax.jit`."""
    jit_client_init = jax.jit(client_init)
    jit_client_step = jax.jit(client_step)
    jit_client_final = jax.jit(client_final)

    def run(shared_input, clients):
      for client_id, client_batches, client_input in clients:
        step_results = []
        state = jit_client_init(shared_input, client_input)
        for batch in client_batches:
          state, step_result = jit_client_step(state, batch)
          step_results.append(step_result)
        output = jit_client_final(shared_input, state)
        yield client_id, output, step_results

    return run


class ForEachClientError(Exception):  # pylint: disable=g-bad-exception-name

  def __init__(self, base: Exception, stage: str, **context):
    super().__init__()
    self.base = base
    self.stage = stage
    self.context = context

  def __str__(self):
    return (f'Stage: {self.stage}. '
            f'Base error is {type(self.base).__name__}: {self.base}.\n'
            'See the `context` field of this exception for additional context.')


class ForEachClientDebugBackend(ForEachClientBackend):
  """for_each_client backend useful during debugging.

  This backend can provide more information for debugging at the cost of being
  slower. With this backend,

  - jax jit compilation is disabled.
  - Exceptions from client_{init,step_final} are wrapped as ForEachClientError
    with the arguments to these functions in the `context` field.
  - Each client is processed sequentially.
  """

  def __call__(self, client_init: ClientInit, client_step: ClientStep,
               client_final: ClientFinal) -> ForEachClient:
    """Creates a `for_each_client` function useful during debugging."""

    def run(shared_input, clients):
      with jax.disable_jit():
        for client_id, client_batches, client_input in clients:
          step_results = []
          try:
            state = client_init(shared_input, client_input)
          except Exception as e:
            raise ForEachClientError(
                e,
                stage='client_init',
                client_id=client_id,
                client_init=client_init,
                shared_input=shared_input,
                client_input=client_input) from e
          for batch in client_batches:
            try:
              state, step_result = client_step(state, batch)
            except Exception as e:
              raise ForEachClientError(
                  e,
                  stage='client_step',
                  client_id=client_id,
                  client_step=client_step,
                  state=state,
                  batch=batch) from e
            step_results.append(step_result)
          try:
            output = client_final(shared_input, state)
          except Exception as e:
            raise ForEachClientError(
                e,
                stage='client_final',
                client_id=client_id,
                client_final=client_final,
                shared_input=shared_input,
                state=state) from e
          yield client_id, output, step_results

    return run


@dataclasses.dataclass
class ClientBlock:
  """Holds data of n clients with padding on both the client and batch levels.

  A ClientBlock holds n uniformly shaped clients, some of which can be "padding
  clients". Further, the final few batches of a client can be "padding batches".

  Attributes:
    client_id: Client ids of each client, or None for a padding client.
    client_mask: Whether each client is a non-padding client.
    num_batches: The number of non-padding batches for each client.
    masked_batches: Uniformly shaped, masked batches for each client. Each list
      of batches for a client has the same length, where the final few batches
      may be padding batches, marked with mask==False.
    client_input: Uniformly shaped client input for each client.
  """
  client_id: List[Optional[ClientId]]
  client_mask: List[bool]
  num_batches: List[int]
  masked_batches: List[List[Tuple[BatchExample, bool]]]
  client_input: List[ClientInput]


def _blockify(clients: Iterable[Tuple[ClientId, Iterable[BatchExample],
                                      ClientInput]],
              block_size: int) -> Iterator[ClientBlock]:
  """Reformats clients into blocks for parallelization.

  Args:
    clients: Iterable of clients, each being a (client_id, client_batches,
      client_input) tuple.
    block_size: Size of output blocks.

  Yields:
    ClientBlocks. The order of clients may be changed in order to minimize the
    amount of padding.
  """
  clients = [(client_id, list(client_batches), client_input)
             for client_id, client_batches, client_input in clients]
  clients.sort(key=lambda x: len(x[1]), reverse=True)
  for i in range(0, len(clients), block_size):
    block = clients[i:i + block_size]
    client_mask = [True for _ in block]
    # Pad to size n.
    if len(block) < block_size:
      _, _, client_input_template = block[0]
      padding_client_input = jax.tree_util.tree_map(jnp.zeros_like,
                                                    client_input_template)
      for j in range(block_size - len(block)):
        # Pad an empty client.
        block.append((None, [], padding_client_input))
        client_mask.append(False)
    # Pad to a fixed number of batches.
    num_batches = [len(client_batches) for _, client_batches, _ in block]
    # Clients are already sorted by decreasing num_batches.
    max_num_batches = num_batches[0]
    masked_batches = []
    if max_num_batches > 0:
      # Use the 0-th batch of the 0-th client in this block as a template.
      batch_template = block[0][1][0]
      padding_batch = jax.tree_util.tree_map(jnp.zeros_like, batch_template)
      for j in range(max_num_batches):
        block_batch = []
        batch_mask = []
        for _, batches, _ in block:
          if j < len(batches):
            block_batch.append(batches[j])
            batch_mask.append(True)
          else:
            block_batch.append(padding_batch)
            batch_mask.append(False)
        masked_batches.append((block_batch, batch_mask))
    yield ClientBlock(
        client_id=[client_id for client_id, _, _ in block],
        client_mask=client_mask,
        num_batches=num_batches,
        masked_batches=masked_batches,
        client_input=[client_input for _, _, client_input in block])


class ForEachClientPmapBackend(ForEachClientBackend):
  """for_each_client backend using jax.pmap for parallelization."""

  def __init__(self, devices: Optional[Sequence[Any]] = None):
    """Initializes a pmap backend.

    Args:
      devices: jax devices to use, defaults to jax.local_devices().
    """
    self._devices = devices

  def __call__(self, client_init: ClientInit, client_step: ClientStep,
               client_final: ClientFinal) -> ForEachClient:
    """Creates a `for_each_client` function using jax.pmap for parallelization.

    Args:
      client_init: ClientInit function.
      client_step: ClientStep function.
      client_final: ClientFinal function.

    Returns:
      Constructed ForEachClient function.
    """
    if self._devices is None:
      devices = jax.local_devices()
    else:
      devices = self._devices
    block_size = len(devices)

    p_client_init = jax.pmap(client_init, in_axes=(None, 0))

    @jax.pmap
    def p_client_step(state, batch, mask):
      next_state, step_result = client_step(state, batch)
      next_state = jax.tree_util.tree_multimap(
          functools.partial(jnp.where, mask), next_state, state)
      step_result = jax.tree_util.tree_map(
          lambda x: jnp.where(mask, x, jnp.zeros_like(x)), step_result)
      return next_state, step_result

    p_client_final = jax.pmap(client_final, in_axes=(None, 0))

    def run(shared_input, clients):
      for block in _blockify(clients, block_size):
        p_state = p_client_init(
            shared_input, jax.device_put_sharded(block.client_input, devices))
        p_step_results = []
        for p_batch, p_mask in block.masked_batches:
          p_state, p_step_result = p_client_step(
              p_state, jax.device_put_sharded(p_batch, devices),
              jax.device_put_sharded(p_mask, devices))
          p_step_results.append(p_step_result)
        p_client_output = p_client_final(shared_input, p_state)
        for i in range(len(block.client_id)):
          if not block.client_mask[i]:
            continue
          client_output, step_results = jax.tree_util.tree_map(
              lambda x: x[i],  # pylint: disable=cell-var-from-loop
              (p_client_output, p_step_results))
          yield (block.client_id[i], client_output,
                 step_results[:block.num_batches[i]])

    return run


class BackendChoice(threading.local):
  """Thread local configuration of the for_each_client backend choice."""

  DEFAULT_BACKEND = ForEachClientJitBackend()

  def __init__(self):
    super().__init__()
    self.backend = None

  def get(self):
    if self.backend is None:
      self.backend = self.DEFAULT_BACKEND
    return self.backend


_BACKEND_CHOICE = BackendChoice()


def get_for_each_client_backend() -> ForEachClientBackend:
  return _BACKEND_CHOICE.get()


def set_for_each_client_backend(backend: Union[ForEachClientBackend, str,
                                               None]):
  """Sets the for_each_client backend for the current thread.

  Args:
    backend: One of the following,

      - None: uses the default backend for the current environment.
      - 'debug': uses the debugging backend.
      - 'jit': uses the JIT backend.
      - 'pmap': uses the pmap-based backend.
      - A concrete ForEachClientBackend object.
  """
  if backend is None or isinstance(backend, ForEachClientBackend):
    _BACKEND_CHOICE.backend = backend
  elif backend == 'debug':
    _BACKEND_CHOICE.backend = ForEachClientDebugBackend()
  elif backend == 'jit':
    _BACKEND_CHOICE.backend = ForEachClientJitBackend()
  elif backend == 'pmap':
    _BACKEND_CHOICE.backend = ForEachClientPmapBackend()
  else:
    raise ValueError(f'Unsupported backend {backend!r}')


@contextlib.contextmanager
def for_each_client_backend(backend: Union[ForEachClientBackend, str, None]):
  """A context manager for switching to a given ForEachClientBackend in the current thread.

  Example::

    with for_each_client_backend('pmap'):
      # We will be using the pmap based for_each_client backend within this block.
      pass
    # We will be using the default for_each_client backend from now on.

  Args:
    backend: See :func:`set_for_each_client_backend`.

  Yields:
    Nothing.
  """
  old = _BACKEND_CHOICE.backend
  try:
    set_for_each_client_backend(backend)
    yield
  finally:
    set_for_each_client_backend(old)


# We leave the return type unannotated because there's no easy way to properly
# annotate it when it depends on the input value of with_step_result.
def for_each_client(client_init: ClientInit,
                    client_step: Union[ClientStep,
                                       Callable[[ClientStepState, BatchExample],
                                                ClientStepState]],
                    client_final: ClientFinal = lambda _, s: s,
                    with_step_result: bool = False):
  """Creates a function which maps over clients.

  For example, for_each_client could be used to define how to run client
  updates for each client in a federated training round. Another common use case
  of for_each_client is to run evaluation per client for a given set of model
  parameters.

  The underlying backend for for_each_client is customizable.
  For example, if multiple devies are available (e.g. TPU), a :func:`jax.pmap`
  based backend can be used to parallelize across devices.
  It's also possible to manually specify which backend to use (for debugging).

  The expected usage of for_each_client is as follows::

    # Map over clients and count how many points are greater than `limit` for
    # each client. Each client also has a different `start` that is specified
    # via client input.

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

    func = fedjax.for_each_client.for_each_client(
        client_init, client_step, client_final)
    print(list(func(shared_input, clients)))
    # [(b'cid0', 5), (b'cid1', 3), (b'cid2', 1)]

  Here's the same example with per step results. ::

    # We'll also keep track of the `num` per step in our step results.

    def client_step_with_result(client_step_state, batch):
      num = jnp.sum(batch['x'] > client_step_state['limit'])
      client_step_state = {
          'limit': client_step_state['limit'],
          'count': client_step_state['count'] + num
      }
      client_step_result = {'num': num}
      return client_step_state, client_step_result

    func = fedjax.for_each_client.for_each_client(
        client_init, client_step_with_result, client_final, with_step_result=True)
    print(list(func(shared_input, clients)))
    # [
    #   (b'cid0', 5, [{'num': 2}, {'num': 1}]),
    #   (b'cid1', 3, [{'num': 0}, {'num': 3}]),
    #   (b'cid2', 1, [{'num': 0}]),
    # ]

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
    A for each client function that takes shared_input and the per client inputs
    as tuple (client_id, batched_client_data, client_rng) to map over and
    returns the outputs per client as specified in `client_final` along with
    optional per client per step results.
  """
  for_each_client_backend_ = get_for_each_client_backend()
  if with_step_result:
    return for_each_client_backend_(client_init, client_step, client_final)

  def client_step_with_result(client_step_state, batch):
    return client_step(client_step_state, batch), ()

  func = for_each_client_backend_(client_init, client_step_with_result,
                                  client_final)

  def run(shared_input, clients):
    for client_id, client_output, _ in func(shared_input, clients):
      yield client_id, client_output

  return run
