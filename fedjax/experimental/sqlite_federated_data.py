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
"""FederatedData backed by SQLite."""

import random
from typing import Callable, Iterable, Iterator, Optional, Tuple
import zlib

from fedjax.experimental import client_datasets
from fedjax.experimental import federated_data
from fedjax.experimental import serialization
import sqlite3


def decompress_and_deserialize(data: bytes):
  data = zlib.decompress(data)
  return serialization.msgpack_deserialize(data)


class SQLiteFederatedData(federated_data.FederatedData):
  """Federated dataset backed by SQLite.

  The SQLite database should contain a table named "federated_data" created with
  the following command:
  ```
  CREATE TABLE federated_data (
    client_id BLOB NOT NULL PRIMARY KEY,
    data BLOB NOT NULL,
    num_examples INTEGER NOT NULL
  );
  ```
  where,
  - `client_id` is the bytes client id.
  - `data` is the serialized client dataset examples.
  - `num_examples` is the number of examples in the client dataset.

  By default we use zlib compressed msgpack blobs for `data` (see
  decompress_and_deserialize()).
  """

  @staticmethod
  def new(
      path: str,
      parse_examples: Callable[
          [bytes], client_datasets.Examples] = decompress_and_deserialize
  ) -> 'SQLiteFederatedData':
    """Opens a federated dataset stored as an SQLite3 database.

    Args:
      path: Path to the SQLite database file.
      parse_examples: Function for deserializing client dataset examples.

    Returns:
      SQLite3DataSource.
    """
    connection = sqlite3.connect(path)
    return SQLiteFederatedData(connection, parse_examples)

  def __init__(self,
               connection: sqlite3.Connection,
               parse_examples: Callable[[bytes], client_datasets.Examples],
               start: Optional[federated_data.ClientId] = None,
               stop: Optional[federated_data.ClientId] = None,
               preprocess_client: client_datasets.Preprocessor = client_datasets
               .NoOpPreprocessor,
               preprocess_batch: client_datasets.Preprocessor = client_datasets
               .NoOpPreprocessor):
    self._connection = connection
    self._parse_examples = parse_examples
    self._start = start
    self._stop = stop
    self._preprocess_client = preprocess_client
    self._preprocess_batch = preprocess_batch

  def slice(
      self,
      start: Optional[federated_data.ClientId] = None,
      stop: Optional[federated_data.ClientId] = None) -> 'SQLiteFederatedData':
    return SQLiteFederatedData(self._connection, self._parse_examples, start,
                               stop, self._preprocess_client,
                               self._preprocess_batch)

  def preprocess_client(
      self, fn: Callable[[client_datasets.Examples], client_datasets.Examples]
  ) -> 'SQLiteFederatedData':
    return SQLiteFederatedData(self._connection, self._parse_examples,
                               self._start, self._stop,
                               self._preprocess_client.append(fn),
                               self._preprocess_batch)

  def preprocess_batch(
      self, fn: Callable[[client_datasets.Examples], client_datasets.Examples]
  ) -> 'SQLiteFederatedData':
    return SQLiteFederatedData(self._connection, self._parse_examples,
                               self._start, self._stop, self._preprocess_client,
                               self._preprocess_batch.append(fn))

  def _range_where(self) -> str:
    """Builds appropriate WHERE clauses for start/stop ranges."""
    if self._start is None and self._stop is None:
      return '(1)'
    elif self._start is not None and self._stop is not None:
      return '(:start <= client_id AND client_id < :stop)'
    elif self._start is None:
      return '(client_id < :stop)'
    else:
      return '(:start <= client_id)'

  def num_clients(self) -> int:
    cursor = self._connection.execute(
        f'SELECT COUNT(*) FROM federated_data WHERE {self._range_where()};', {
            'start': self._start,
            'stop': self._stop
        })
    return cursor.fetchone()[0]

  def client_ids(self) -> Iterator[federated_data.ClientId]:
    cursor = self._connection.execute(
        f'SELECT client_id FROM federated_data WHERE {self._range_where()} ORDER BY rowid;',
        {
            'start': self._start,
            'stop': self._stop
        })
    while True:
      result = cursor.fetchone()
      if result is None:
        break
      yield result[0]

  def client_sizes(self) -> Iterator[Tuple[federated_data.ClientId, int]]:
    cursor = self._connection.execute(
        f'SELECT client_id, num_examples FROM federated_data WHERE {self._range_where()} ORDER BY rowid;',
        {
            'start': self._start,
            'stop': self._stop
        })
    while True:
      result = cursor.fetchone()
      if result is None:
        break
      yield tuple(result)

  def client_size(self, client_id: federated_data.ClientId) -> int:
    if ((self._start is None or self._start <= client_id) and
        (self._stop is None or client_id < self._stop)):
      cursor = self._connection.execute(
          'SELECT num_examples FROM federated_data WHERE client_id = ?',
          [client_id])
      result = cursor.fetchone()
      if result is not None:
        return result[0]
    raise KeyError

  def clients(
      self
  ) -> Iterator[Tuple[federated_data.ClientId, client_datasets.ClientDataset]]:
    for k, v in self._read_clients():
      yield k, self._client_dataset(v)

  def _read_clients(self):
    cursor = self._connection.execute(
        f'SELECT client_id, data FROM federated_data WHERE {self._range_where()} ORDER BY rowid;',
        {
            'start': self._start,
            'stop': self._stop
        })
    while True:
      result = cursor.fetchone()
      if result is None:
        break
      yield tuple(result)

  def shuffled_clients(
      self,
      buffer_size: int,
      seed: Optional[int] = None
  ) -> Iterator[Tuple[federated_data.ClientId, client_datasets.ClientDataset]]:
    rng = random.Random(seed)
    while True:
      for k, v in federated_data.buffered_shuffle(self._read_clients(),
                                                  buffer_size, rng):
        yield k, self._client_dataset(v)

  def get_clients(
      self, client_ids: Iterable[federated_data.ClientId]
  ) -> Iterator[Tuple[federated_data.ClientId, client_datasets.ClientDataset]]:
    for client_id in client_ids:
      yield client_id, self.get_client(client_id)

  def get_client(
      self,
      client_id: federated_data.ClientId) -> client_datasets.ClientDataset:
    if ((self._start is None or self._start <= client_id) and
        (self._stop is None or client_id < self._stop)):
      cursor = self._connection.execute(
          'SELECT data FROM federated_data WHERE client_id = ?', [client_id])
      result = cursor.fetchone()
      if result is not None:
        return self._client_dataset(result[0])
    raise KeyError

  def _client_dataset(self, v: bytes) -> client_datasets.ClientDataset:
    examples = self._preprocess_client(self._parse_examples(v))
    return client_datasets.ClientDataset(examples, self._preprocess_batch)
