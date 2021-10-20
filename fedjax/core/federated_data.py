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
from typing import Any, Callable, Iterable, Iterator, Optional, Tuple

from fedjax.core import client_datasets
import numpy as np

# A client id is simply some binary bytes.
ClientId = bytes


class ClientPreprocessor:
  """A chain of preprocessing functions on all examples of a client dataset.

  This is very similar to :class:`fedjax.BatchPreprocessor`, with the main
  difference being that ClientPreprocessor also takes ``client_id`` as input.

  See the discussion in :class:`fedjax.BatchPreprocessor` regarding when to
  use which.
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

  A FederatedData object serves as a mapping from client ids to client
  datasets and client metadata.

  **Access methods with better I/O efficiency**

  For large federated datasets, it is not feasible to load all client datasets
  into memory at once (whereas loading a single client dataset is assumed to be
  feasible). Different implementations exist for different on disk storage
  formats. Since sequential read is much faster than random read for most
  storage technologies, FederatedData provides two types of methods for
  accessing client datasets,

  1.  :meth:`clients` and :meth:`shuffled_clients` are sequential read
      friendly, and thus recommended whenever appropriate.
  2.  :meth:`get_clients` requires random read, but prefetching is possible.
      This should be preferred over :meth:`get_client`.
  3.  :meth:`get_client` is usually the slowest way of accessing client
      datasets, and is mostly intended for interactive exploration of a small
      number of clients.

  **Preprocessing**

  :class:`~fedjax.ClientDataset` produced by FederatedData can hold a
  :class:`~fedjax.BatchPreprocessor`, customizable via :meth:`preprocess_batch`.
  Additionally, another "client" level :class:`ClientPreprocessor`, customizable
  via :meth:`preprocess_client`, can be used to apply transformations on
  examples from the entire client dataset before a
  :class:`~fedjax.ClientDataset` is constructed.
  """

  @abc.abstractmethod
  def slice(self,
            start: Optional[ClientId] = None,
            stop: Optional[ClientId] = None) -> 'FederatedData':
    """Returns a new FederatedData restricted to client ids in the given range.

    The returned FederatedData includes clients whose ids are,

    - Greater than or equal to ``start`` when ``start`` is not None;
    - Less than ``stop`` when ``stop`` is not None.

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
    """Returns an iterator of client ids as bytes.

    There is no requirement on the order of iteration.
    """

  @abc.abstractmethod
  def client_sizes(self) -> Iterator[Tuple[ClientId, int]]:
    """Returns an iterator of all (client id, client size) pairs.

    This is often more efficient than making multiple :meth:`client_size` calls.
    There is no requirement on the order of iteration.
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

    Shuffling should use a buffer size of at least ``buffer_size`` clients.
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

    Clients are returned in the order of ``client_ids``.

    Args:
      client_ids: Client ids to load.

    Returns:
      Iterator.
    """

  @abc.abstractmethod
  def get_client(self, client_id: ClientId) -> client_datasets.ClientDataset:
    """Gets one single client dataset.

    Prefer :meth:`clients`, :meth:`shuffled_clients`, or :meth:`get_clients`
    when possible.

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


# Functions for treating a federated dataset as a single centralized dataset.


def shuffle_repeat_batch_federated_data(
    fd: FederatedData,
    batch_size: int,
    client_buffer_size: int,
    example_buffer_size: int,
    seed: Optional[int] = None) -> Iterator[client_datasets.Examples]:
  """Shuffle-repeat-batch all client datasets in a federated dataset for training a centralized baseline.

  Shuffling is done using two levels of buffered shuffling, first at the client
  level, then at the example level.

  This produces an infinite stream of batches. itertools.islice() can be used to
  cap the number of batches, if so desired.

  Args:
    fd: Federated dataset.
    batch_size: Desired batch size.
    client_buffer_size: Buffer size for client level shuffling.
    example_buffer_size: Buffer size for example level shuffling.
    seed: Optional RNG seed.

  Yields:
    Batches of preprocessed examples.
  """
  rng = np.random.RandomState(seed)
  datasets = (client_dataset for _, client_dataset in fd.shuffled_clients(
      client_buffer_size, rng.randint(1 << 32)))
  yield from client_datasets.buffered_shuffle_batch_client_datasets(
      datasets, batch_size=batch_size, buffer_size=example_buffer_size, rng=rng)


def padded_batch_federated_data(fd: FederatedData,
                                hparams: Optional[
                                    client_datasets.PaddedBatchHParams] = None,
                                **kwargs) -> Iterator[client_datasets.Examples]:
  """Padded batch all client datasets, useful for evaluation on the entire federated dataset.

  Args:
    fd: Federated dataset.
    hparams: See :func:`fedjax.padded_batch_client_datasets`.
    **kwargs: See :func:`fedjax.padded_batch_client_datasets`.

  Yields:
    Batches of preprocessed examples.
  """
  datasets = (client_dataset for _, client_dataset in fd.clients())
  yield from client_datasets.padded_batch_client_datasets(
      datasets, hparams, **kwargs)


def intersect_slice_ranges(
    current_start: Optional[ClientId], current_stop: Optional[ClientId],
    new_start: Optional[ClientId], new_stop: Optional[ClientId]
) -> Tuple[Optional[ClientId], Optional[ClientId]]:
  """Intersects the current slice range and the new slice range.

  This is a helper function for FederatedData implementations for ensuring
  slicing does not enlarge the range of client ids.

  Args:
    current_start: Current start of the slice range.
    current_stop: Current stop of the slice range.
    new_start: New start of the slice range.
    new_stop: New stop of the slice range.

  Returns:
    Normalized slice range that is the intersection of the two input ranges.
  """
  if current_start is not None:
    if new_start is None:
      new_start = current_start
    else:
      new_start = max(current_start, new_start)
  if current_stop is not None:
    if new_stop is None:
      new_stop = current_stop
    else:
      new_stop = min(current_stop, new_stop)
  return new_start, new_stop


class RepeatableIterator:
  """Repeats a base iterable after the end of the first pass of iteration.

  Because this is a stateful object, it is not thread safe, and all usual
  caveats about accessing the same iterator at different locations apply. For
  example, if we make two map calls to the same RepeatableIterator, we must make
  sure we do not interleave `next()` calls on these. For example, the following
  is safe because we finish iterating on m1 before starting to iterate on m2., ::

    it = RepeatableIterator(range(4))
    m1 = map(lambda x: x + 1, it)
    m2 = map(lambda x: x * x, it)
    # We finish iterating on m1 before starting to iterate on m2.
    print(list(m1), list(m2))
    # [1, 2, 3, 4] [0, 1, 4, 9]

  Whereas interleaved access leads to confusing results, ::

    it = RepeatableIterator(range(4))
    m1 = map(lambda x: x + 1, it)
    m2 = map(lambda x: x * x, it)
    print(next(m1), next(m2))
    # 1 1
    print(next(m1), next(m2))
    # 3 9
    print(next(m1), next(m2))
    # StopIteration!

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


class SubsetFederatedData(FederatedData):
  """A simple wrapper over a concrete FederatedData for restricting to a subset of client ids.

  This is useful when we wish to create a smaller FederatedData out of arbitrary
  client ids, where slicing is not possible.
  """

  def __init__(self,
               base: FederatedData,
               client_ids: Iterable[ClientId],
               validate=True):
    """Initializes the subset federated dataset.

    Args:
      base: Base concrete FederatedData.
      client_ids: Client ids to include in the subset. All client ids must be in
        base.client_ids(), otherwise the behavior of SubsetFederatedData is
        undefined when validate=False.
      validate: Whether to validate client ids.
    """
    self._base = base
    if not isinstance(client_ids, set):
      client_ids = set(client_ids)
    if validate:
      bad_client_ids = client_ids.difference(base.client_ids())
      if bad_client_ids:
        raise ValueError('Some client ids are not in the base FederatedData, '
                         f'showing up to 10: {sorted(bad_client_ids)[:10]}')
    self._client_ids = client_ids

  def slice(self,
            start: Optional[ClientId] = None,
            stop: Optional[ClientId] = None) -> FederatedData:
    if start is None and stop is None:
      client_ids = self._client_ids
    elif start is None:
      client_ids = set(i for i in self._client_ids if i < stop)
    elif stop is None:
      client_ids = set(i for i in self._client_ids if i >= start)
    else:
      client_ids = set(i for i in self._client_ids if start <= i and i < stop)
    return SubsetFederatedData(
        self._base.slice(start, stop), client_ids, validate=False)

  def num_clients(self) -> int:
    return len(self._client_ids)

  def client_ids(self) -> Iterator[ClientId]:
    # Ids are sorted for deterministic iteration order.
    return iter(sorted(self._client_ids))

  def client_sizes(self) -> Iterator[Tuple[ClientId, int]]:
    for client_id, size in self._base.client_sizes():
      if client_id in self._client_ids:
        yield client_id, size

  def client_size(self, client_id: ClientId) -> int:
    if client_id not in self._client_ids:
      raise KeyError
    return self._base.client_size(client_id)

  def clients(self) -> Iterator[Tuple[ClientId, client_datasets.ClientDataset]]:
    # Ids are sorted for deterministic iteration order.
    yield from self.get_clients(sorted(self._client_ids))

  def shuffled_clients(
      self,
      buffer_size: int,
      seed: Optional[int] = None
  ) -> Iterator[Tuple[ClientId, client_datasets.ClientDataset]]:
    rng = np.random.RandomState(seed)
    while True:
      for client_id, dataset in client_datasets.buffered_shuffle(
          self.clients(), buffer_size, rng):
        yield client_id, dataset

  def get_clients(
      self, client_ids: Iterable[ClientId]
  ) -> Iterator[Tuple[ClientId, client_datasets.ClientDataset]]:
    for client_id, dataset in self._base.get_clients(client_ids):
      if client_id not in self._client_ids:
        raise KeyError
      yield client_id, dataset

  def get_client(self, client_id: ClientId) -> client_datasets.ClientDataset:
    if client_id not in self._client_ids:
      raise KeyError
    return self._base.get_client(client_id)

  def preprocess_client(
      self, fn: Callable[[ClientId, client_datasets.Examples],
                         client_datasets.Examples]
  ) -> FederatedData:
    return SubsetFederatedData(
        self._base.preprocess_client(fn), self._client_ids, validate=False)

  def preprocess_batch(
      self, fn: Callable[[client_datasets.Examples], client_datasets.Examples]
  ) -> FederatedData:
    return SubsetFederatedData(
        self._base.preprocess_batch(fn), self._client_ids, validate=False)


class FederatedDataBuilder(abc.ABC):
  """FederatedDataBuilder interface.

  To be implemented as a context manager for building file formats from pairs of
  client IDs and client NumPy examples.

  It is relevant to note that the add method below does not specify any raised
  exceptions. One could imagine some formats where add can fail in some way:
  out-of-order or duplicate inputs, remote files and network failures,
  individual entries too big for a format, etc. In order to address this we let
  implementations throw whatever they see relevant and fit to their particular
  use cases. The same is relevant when it comes to the __init__, __enter__, and
  __exit__ methods, where implementations are left with the responsibility of
  raising exceptions as they see fit to their particular use cases. For example,
  if an invalid file path is passed, or there were any issues finalizing the
  builder, etc.

  Eg of end behavior::

    with FederatedDataBuilder(path) as builder:
      builder.add(b'k1', np.array([b'v1'], dtype=np.object))
      builder.add(b'k2', np.array([b'v2'], dtype=np.object))
  """

  @abc.abstractmethod
  def __enter__(self):
    """Assigns the variable defined after 'as' in the with statement to self.

    By returning self the required functionality is kept within the same class
    so that one can call the add method defined below inside the with block.

    Returns:
      self
    """

  @abc.abstractmethod
  def __exit__(self, exc_type, exc_value, exc_traceback):
    """Finalizes the builder once it leaves the 'with' block.

    Args:
      exc_type: indicates class of exception.
      exc_value: indicates type of exception.
      exc_traceback: traceback is a report which has all of the information
        needed to solve the exception.
    """

  @abc.abstractmethod
  def add_many(self,
               client_ids_examples: Iterable[Tuple[bytes,
                                                   client_datasets.Examples]]):
    """Bulk adds multiple client IDs and client NumPy examples pairs to file format.

    Args:
      client_ids_examples: Iterable of tuples of client id and examples.
    """
