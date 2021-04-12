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

  `ClientDataset`s produced by `FederatedData` can hold a `Preprocessor`,
  customizable via `preprocess_batch()`. Additionally, another "client" level
  `Preprocessor`, customizable via `preprocess_client()`, can be used to apply
  transformations on examples from the entire client dataset before a
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
      self, fn: Callable[[client_datasets.Examples], client_datasets.Examples]
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
