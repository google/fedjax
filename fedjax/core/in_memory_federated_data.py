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
"""InMemoryFederatedData for small custom datasets."""

from typing import Callable, Iterable, Iterator, Mapping, Optional, Tuple

from fedjax.core import client_datasets
from fedjax.core import federated_data
import numpy as np


class InMemoryFederatedData(federated_data.FederatedData):
  """A simple wrapper over a concrete fedjax.FederatedData for small in memory datasets.

  This is useful when we wish to create a smaller FederatedData that fits in
  memory. Here is a simple example to create a fedjax.InMemoryFederatedData, ::

    client_a_data = {
        'x': np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        'y': np.array([7, 8])
    }
    client_b_data = {'x': np.array([[9.0, 10.0, 11.0]]), 'y': np.array([12])}
    client_to_data_mapping = {'a': client_a_data, 'b': client_b_data}

    fedjax.InMemoryFederatedData(client_to_data_mapping)

  Returns:
    A fedjax.InMemoryDataset corresponding to client_to_data_mapping.
  """

  def __init__(self,
               client_to_data_mapping: Mapping[federated_data.ClientId,
                                               Mapping[str, np.ndarray]],
               preprocess_client: federated_data
               .ClientPreprocessor = federated_data.NoOpClientPreprocessor,
               preprocess_batch: client_datasets
               .BatchPreprocessor = client_datasets.NoOpBatchPreprocessor):
    """Initializes the in memory federated dataset.

      Data of each client is a mapping from feature names to numpy arrays. For
      example, for emnist image classification,
      {'x': X, 'y': y}, where X is a matrix of shape (num_data_points, 28, 28)
      and y is a matrix of shape (num_data_points).

    Args:
      client_to_data_mapping: A mapping from client_id to data of each client.
      preprocess_client: federated_data.ClientPreprocessor to preprocess each
        client data.
      preprocess_batch: client_datasets.BatchPreprocessor to preprocess batch of
        data.
    """
    self._preprocess_client = preprocess_client
    self._preprocess_batch = preprocess_batch
    self._client_to_data_mapping = client_to_data_mapping
    self._client_ids = sorted(self._client_to_data_mapping.keys())
    self._features = list(
        self._client_to_data_mapping[self._client_ids[0]].keys())
    for client_id in self._client_ids:
      dataset = self._client_to_data_mapping[client_id]
      if list(dataset.keys()) != self._features:
        raise ValueError(
            f'Inconsistent features, got {list(dataset.keys())} for client {client_id}, expect {self._features}'
        )

      num_samples = dataset[self._features[0]].shape[0]
      try:
        client_datasets.assert_consistent_rows(
            self._client_dataset(client_id).all_examples())
      except ValueError as exc:
        raise ValueError(
            f'Inconsistent examples, for client {client_id}') from exc

  def slice(
      self,
      start: Optional[federated_data.ClientId] = None,
      stop: Optional[federated_data.ClientId] = None
  ) -> federated_data.FederatedData:
    if start is None and stop is None:
      client_ids = self._client_ids
    elif start is None:
      client_ids = set(i for i in self._client_ids if i < stop)
    elif stop is None:
      client_ids = set(i for i in self._client_ids if i >= start)
    else:
      client_ids = set(i for i in self._client_ids if start <= i and i < stop)
    return InMemoryFederatedData(
        {
            client_id: self._client_to_data_mapping[client_id]
            for client_id in client_ids
        }, self._preprocess_client, self._preprocess_batch)

  def num_clients(self) -> int:
    return len(self._client_ids)

  def client_ids(self) -> Iterator[federated_data.ClientId]:
    # Ids are sorted for deterministic iteration order.
    return iter(sorted(self._client_ids))

  def client_sizes(self) -> Iterator[Tuple[federated_data.ClientId, int]]:
    for client_id in self._client_ids:
      yield client_id, client_datasets.num_examples(
          self._client_dataset(client_id).all_examples(), validate=False)

  def client_size(self, client_id: federated_data.ClientId) -> int:
    return client_datasets.num_examples(
        self._client_dataset(client_id).all_examples(), validate=False)

  def clients(
      self
  ) -> Iterator[Tuple[federated_data.ClientId, client_datasets.ClientDataset]]:
    # Ids are sorted for deterministic iteration order.
    yield from self.get_clients(self._client_ids)

  def shuffled_clients(
      self,
      buffer_size: int,
      seed: Optional[int] = None
  ) -> Iterator[Tuple[federated_data.ClientId, client_datasets.ClientDataset]]:
    rng = np.random.RandomState(seed)
    while True:
      for client_id, dataset in client_datasets.buffered_shuffle(
          self.clients(), buffer_size, rng):
        yield client_id, dataset

  def get_clients(
      self, client_ids: Iterable[federated_data.ClientId]
  ) -> Iterator[Tuple[federated_data.ClientId, client_datasets.ClientDataset]]:
    for client_id in client_ids:
      yield client_id, self._client_dataset(client_id)

  def get_client(
      self,
      client_id: federated_data.ClientId) -> client_datasets.ClientDataset:
    return self._client_dataset(client_id)

  def preprocess_client(
      self, fn: Callable[[federated_data.ClientId, client_datasets.Examples],
                         client_datasets.Examples]
  ) -> federated_data.FederatedData:
    return InMemoryFederatedData(self._client_to_data_mapping,
                                 self._preprocess_client.append(fn),
                                 self._preprocess_batch)

  def preprocess_batch(
      self, fn: Callable[[client_datasets.Examples], client_datasets.Examples]
  ) -> federated_data.FederatedData:
    return InMemoryFederatedData(self._client_to_data_mapping,
                                 self._preprocess_client,
                                 self._preprocess_batch.append(fn))

  def _client_dataset(
      self,
      client_id: federated_data.ClientId) -> client_datasets.ClientDataset:
    examples = self._preprocess_client(client_id,
                                       self._client_to_data_mapping[client_id])
    return client_datasets.ClientDataset(examples, self._preprocess_batch)
