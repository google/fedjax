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

from typing import Callable, Iterable, Iterator, Optional, Tuple, List
import zlib

from fedjax.core import client_datasets
from fedjax.core import federated_data
from fedjax.core import serialization
import numpy as np
import sqlite3


def decompress_and_deserialize(data: bytes):
  data = zlib.decompress(data)
  return serialization.msgpack_deserialize(data)


class SQLiteFederatedData(federated_data.FederatedData):
  """Federated dataset backed by SQLite.

  The SQLite database should contain a table named "federated_data" created with
  the following command::

    CREATE TABLE federated_data (
      client_id BLOB NOT NULL PRIMARY KEY,
      data BLOB NOT NULL,
      num_examples INTEGER NOT NULL
    );

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
               preprocess_client: federated_data
               .ClientPreprocessor = federated_data.NoOpClientPreprocessor,
               preprocess_batch: client_datasets
               .BatchPreprocessor = client_datasets.NoOpBatchPreprocessor):
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
    start, stop = federated_data.intersect_slice_ranges(self._start, self._stop,
                                                        start, stop)
    return SQLiteFederatedData(self._connection, self._parse_examples, start,
                               stop, self._preprocess_client,
                               self._preprocess_batch)

  def preprocess_client(
      self, fn: Callable[[federated_data.ClientId, client_datasets.Examples],
                         client_datasets.Examples]
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
      yield k, self._client_dataset(k, v)

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
    rng = np.random.RandomState(seed)
    while True:
      for k, v in client_datasets.buffered_shuffle(self._read_clients(),
                                                   buffer_size, rng):
        yield k, self._client_dataset(k, v)

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
        return self._client_dataset(client_id, result[0])
    raise KeyError

  def _client_dataset(self, client_id: federated_data.ClientId,
                      data: bytes) -> client_datasets.ClientDataset:
    examples = self._preprocess_client(client_id, self._parse_examples(data))
    return client_datasets.ClientDataset(examples, self._preprocess_batch)


class SQLiteFederatedDataBuilder(federated_data.FederatedDataBuilder):
  """Builds SQLite files from a python dictionary containing an arbitrary mapping of client IDs to NumPy examples."""

  def __init__(self, path: str):
    """Initializes SQLiteBuilder by opening a connection and setting up the database with columns.

    Args:
      path: Path of file to write to (e.g. /tmp/sqlite_federated_data.sqlite).
    """

    self._connection = sqlite3.connect(path)
    self._connection.execute(""" CREATE TABLE federated_data (
    client_id BLOB NOT NULL PRIMARY KEY,
    data BLOB NOT NULL,
    num_examples INTEGER NOT NULL );""")
    self._connection.commit()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, exc_traceback):
    self._connection.close()

  def add_many(self,
               client_ids_examples: Iterable[Tuple[bytes,
                                                   client_datasets.Examples]]):

    def prepare_parameters(ce):
      client_id = ce[0]
      examples = ce[1]
      num_examples = client_datasets.num_examples(examples, validate=True)
      data = zlib.compress(serialization.msgpack_serialize(examples))
      return client_id, data, num_examples

    client_ids_datas_num_examples = map(prepare_parameters, client_ids_examples)
    self._connection.executemany('INSERT INTO federated_data VALUES (?, ?, ?);',
                                 client_ids_datas_num_examples)
    self._connection.commit()


class TFFSQLiteClientsIterator:
  """Iterator over clients stored in TensorFlow Federated (TFF) SQLite tables.

  TFF uses a "1 example per record" format, where each row in the SQLite table
  is a key-value pair from the client id to a protocol buffer message of a
  single example. A custom `parse_examples` function is thus necessary for
  different datasets in order to parse different example formats.

  The TFF. SQLite database should contain two tables named "examples" and
  "client_metadata" created with the following command::

    CREATE TABLE examples (
        split_name TEXT NOT NULL,
        client_id TEXT NOT NULL,
        serialized_example_proto BLOB NOT NULL);

    CREATE INDEX idx_examples_client_id
      ON examples (client_id);

    CREATE INDEX idx_examples_client_id_split
      ON examples (split_name, client_id);

    CREATE TABLE client_metadata (
      client_id TEXT NOT NULL,
      split_name TEXT NOT NULL,
      num_examples INTEGER NOT NULL);

    CREATE INDEX idx_metadata_client_id
      ON client_metadata (client_id);

  where,

  - `split_name` is the split name (e.g. "train" or "test").
  - `client_id` is the client id.
  - `serialized_example_proto` is a single serialized `tf.train.Example`
  protocol buffer message.
  - `num_examples` is the number of examples in the client dataset.
  """

  def __init__(self, path: str,
               parse_examples: Callable[[List[bytes]],
                                        client_datasets.Examples],
               split_name: str):
    self._connection = sqlite3.connect(path)
    self._parse_examples = parse_examples
    self._split_name = split_name
    self._client_ids_cursor = self._connection.execute(
        'SELECT client_id FROM client_metadata WHERE split_name = ? ORDER BY client_id;',
        [split_name])

  def __del__(self):
    self._connection.close()

  def __iter__(
      self
  ) -> Iterator[Tuple[federated_data.ClientId, client_datasets.ClientDataset]]:
    return self

  def __next__(
      self) -> Tuple[federated_data.ClientId, client_datasets.ClientDataset]:
    client_ids_result = self._client_ids_cursor.fetchone()
    if client_ids_result is None:
      raise StopIteration
    client_id = client_ids_result[0]
    examples_cursor = self._connection.execute(
        'SELECT serialized_example_proto FROM examples WHERE split_name = ? AND client_id = ? ORDER BY rowid;',
        [self._split_name, client_id])
    examples = [r[0] for r in examples_cursor.fetchall()]
    return client_id, client_datasets.ClientDataset(
        self._parse_examples(examples))
