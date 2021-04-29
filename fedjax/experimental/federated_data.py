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
"""FederatedData interface for providing access to a federated dataset."""

import abc
import itertools
import random
from typing import Any, Callable, Iterable, Iterator, Optional, Tuple

from fedjax.experimental import client_datasets

# A client id is simply some binary bytes.
ClientId = bytes


class ClientPreprocessor:
  """A chain of preprocessing functions on all examples of a client dataset.

  This is very similar to `client_datasets.BatchPreprocessor`, with the main
  difference being that `ClientPreprocessor` also takes
  `client_id` as input.

  See the discussion in `BatchPreprocessor` regarding when to use which.
  """

  def __init__(self,
               fns: Iterable[Callable[[ClientId, client_datasets.Examples],
                                      client_datasets.Examples]] = ()):
    self._fns = tuple(fns)

  def __call__(self, client_id: ClientId,
               examples: client_datasets.Examples) -> client_datasets.Examples:
    if not self._fns:
      return examples
    # Make a copy to guard against fns that modify their input.
    out = dict(examples)
    for f in self._fns:
      out = f(client_id, out)
    client_datasets.assert_consistent_rows(out)
    return out

  def append(
      self, fn: Callable[[ClientId, client_datasets.Examples],
                         client_datasets.Examples]
  ) -> 'ClientPreprocessor':
    """Creates a new ClientPreprocessor with fn added to the end."""
    return ClientPreprocessor(self._fns + (fn,))

  def __str__(self) -> str:
    return f'ClientPreprocessor({self._fns})'

  def __repr__(self) -> str:
    return str(self)


# A common default preprocessor that does nothing.
NoOpClientPreprocessor = ClientPreprocessor()


class FederatedData(abc.ABC):
  """FederatedData interface for providing access to a federated dataset.

  A `FederatedData` object serves as a mapping from client ids to client
  datasets and client metadata.

  ##  Access methods with better I/O efficiency

  For large federated datasets, it is not feasible to load all client datasets
  into memory at once (whereas loading a single client dataset is assumed to be
  feasible). Different implementations exist for different on disk storage
  formats. Since sequential read is much faster than random read for most
  storage technologies, `FederatedData` provides two types of methods for
  accessing client datasets,

  1.  `clients()` and `shuffled_clients()` are sequential read friendly, and
  thus recommended whenever appropriate.
  2.  `get_clients()` requires random read, but prefetching is possible. This
  should be preferred over `get_client()`.
  3.  `get_client()` is usually the slowest way of accessing client datasets,
  and is mostly intended for interactive exploration of a small number of
  clients.

  ##  Preprocessing

  `ClientDataset`s produced by `FederatedData` can hold a `BatchPreprocessor`,
  customizable via `preprocess_batch()`. Additionally, another "client" level
  `ClientPreprocessor`, customizable via `preprocess_client()`, can be used to
  apply transformations on examples from the entire client dataset before a
  ClientDataset is constructed.
  """

  @abc.abstractmethod
  def slice(self,
            start: Optional[ClientId] = None,
            stop: Optional[ClientId] = None) -> 'FederatedData':
    """Returns a new FederatedData restricted to client ids in the given range.

    The returned FederatedData includes clients whose ids are,
    -   Greater than or equal to `start` when `start` is not None;
    -   Less than `stop` when `stop` is not None.

    Args:
      start: Start of client id range.
      stop: Stop of client id range.

    Returns:
      FederatedData.
    """

  # Client metadata access.

  @abc.abstractmethod
  def num_clients(self) -> int:
    """Returns the number of clients.

    If it is too expensive or otherwise impossible to obtain the result, an
    implementation may raise an exception.
    """

  @abc.abstractmethod
  def client_ids(self) -> Iterator[ClientId]:
    """Returns an iterable of client ids as bytes.

    There is no requirement on the order of iteration.
    """

  @abc.abstractmethod
  def client_sizes(self) -> Iterator[Tuple[ClientId, int]]:
    """Returns an iterable of all (client id, client size) pairs.

    This is often more efficient than making multiple client_size() calls. There
    is no requirement on the order of iteration.
    """

  @abc.abstractmethod
  def client_size(self, client_id: ClientId) -> int:
    """Returns the number of examples in a client dataset."""

  # Client access.

  @abc.abstractmethod
  def clients(self) -> Iterator[Tuple[ClientId, client_datasets.ClientDataset]]:
    """Iterates over clients in a deterministic order.

    Implementation can choose whatever order that makes iteration efficient.
    """

  @abc.abstractmethod
  def shuffled_clients(
      self,
      buffer_size: int,
      seed: Optional[int] = None
  ) -> Iterator[Tuple[ClientId, client_datasets.ClientDataset]]:
    """Iterates over clients with a repeated buffered shuffling.

    Shuffling should use a buffer with a size of at least `buffer_size` clients.
    The iteration should repeat forever, with usually a different order in each
    pass.

    Args:
      buffer_size: Buffer size for shuffling.
      seed: Optional random number generator seed.

    Returns:
      Iterator.
    """

  @abc.abstractmethod
  def get_clients(
      self, client_ids: Iterable[ClientId]
  ) -> Iterator[Tuple[ClientId, client_datasets.ClientDataset]]:
    """Gets multiple clients in order with one call.

    Clients are returned in the order of `client_ids`.

    Args:
      client_ids: Client ids to load.

    Returns:
      Iterator.
    """

  @abc.abstractmethod
  def get_client(self, client_id: ClientId) -> client_datasets.ClientDataset:
    """Gets one single client dataset.

    Prefer clients(), shuffled_clients(), or get_clients() when possible.

    Args:
      client_id: Client id to load.

    Returns:
      The corresponding ClientDataset.
    """

  # Preprocessing.

  @abc.abstractmethod
  def preprocess_client(
      self, fn: Callable[[ClientId, client_datasets.Examples],
                         client_datasets.Examples]
  ) -> 'FederatedData':
    """Registers a preprocessing function to be called on all examples of a client before passing them to construct a ClientDataset."""

  @abc.abstractmethod
  def preprocess_batch(
      self, fn: Callable[[client_datasets.Examples], client_datasets.Examples]
  ) -> 'FederatedData':
    """Registers a preprocessing function to be called after batching in ClientDatasets."""


# Utility functions useful when implementing FederatedData.


def buffered_shuffle(source: Iterable[Any], buffer_size: int,
                     rng: random.Random) -> Iterator[Any]:
  """Shuffles an iterable via buffered shuffling."""
  it = iter(source)
  buf = list(itertools.islice(it, buffer_size))
  rng.shuffle(buf)
  for i in it:
    r, buf[0] = buf[0], i
    swap = rng.randrange(buffer_size)
    if swap < buffer_size - 1:
      buf[swap], buf[0] = buf[0], buf[swap]
    yield r
  for i in buf:
    yield i


class RepeatableIterator:
  """Repeats a base iterable after the end of the first pass of iteration.

  Because this is a stateful object, it is not thread safe, and all usual
  caveats about accessing the same iterator at different locations apply. For
  example, if we make two map calls to the same RepeatableIterator, we must make
  sure we do not interleave `next()` calls on these. For example, the following
  is safe because we finish iterating on m1 before starting to iterate on m2.,
  ```
  it = RepeatableIterator(range(4))
  m1 = map(lambda x: x + 1, it)
  m2 = map(lambda x: x * x, it)
  # We finish iterating on m1 before starting to iterate on m2.
  print(list(m1), list(m2))
  # [1, 2, 3, 4] [0, 1, 4, 9]
  ```
  Whereas interleaved access leads to confusing results,
  ```
  it = RepeatableIterator(range(4))
  m1 = map(lambda x: x + 1, it)
  m2 = map(lambda x: x * x, it)
  print(next(m1), next(m2))
  # 1 1
  print(next(m1), next(m2))
  # 3 9
  print(next(m1), next(m2))
  # StopIteration!
  ```

  In the first pass of iteration, values fetched from the base iterator will be
  copied onto an internal buffer (except for a few builtin containers where
  copying is unnecessary). When each pass of iteration finishes (i.e. when
  __next__() raises StopIteration), the iterator resets itself to the start of
  the buffer, thus allowing a subsequent pass of repeated iteration.

  In most cases, if repeated iterations are required, it is sufficient to simply
  copy values from an iterator into a list. However, sometimes an iterator
  produces values via potentially expensive I/O operations (e.g. loading client
  datasets), RepeatableIterator can interleave I/O and JAX compute to decrease
  accelerator idle time in this case.
  """

  def __init__(self, base: Iterable[Any]):
    if any(
        isinstance(base, container)
        for container in (list, tuple, dict, str, bytes)):
      # No copying for builtin containers that are already repeatable.
      self._first_pass = False
      self._iter = iter(base)
      self._buf = base
    else:
      # General case, copying required.
      self._first_pass = True
      self._iter = iter(base)
      self._buf = []

  def __iter__(self) -> Iterator[Any]:
    return self

  def __next__(self) -> Any:
    try:
      value = next(self._iter)
    except StopIteration:
      if self._first_pass:
        self._first_pass = False
      self._iter = iter(self._buf)
      raise
    if self._first_pass:
      self._buf.append(value)
    return value
