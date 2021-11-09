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
"""Tests for fedjax.core.sqlite_federated_data."""

import itertools
import os.path
import zlib

from absl import flags
from absl.testing import absltest
from fedjax.core import serialization
from fedjax.core import sqlite_federated_data
from fedjax.core import util
import numpy as np
import numpy.testing as npt
import sqlite3

tf = util.import_tf()

FLAGS = flags.FLAGS


class SQLiteFederatedDataTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    path = os.path.join(FLAGS.test_tmpdir, 'test_sqlite_federated_data.sqlite')

    # Create database file. First make sure the database file is empty.
    with open(path, 'w'):
      pass
    connection = sqlite3.connect(path)
    with connection:
      connection.execute("""
      CREATE TABLE federated_data (
        client_id BLOB NOT NULL PRIMARY KEY,
        data BLOB NOT NULL,
        num_examples INTEGER NOT NULL
      );""")
      for i in range(100):
        client_id = f'{i:04x}'.encode('ascii')
        features = {'x': np.arange(i + 1)}
        data = zlib.compress(serialization.msgpack_serialize(features))
        num_examples = i + 1
        connection.execute('INSERT INTO federated_data VALUES (?, ?, ?);',
                           [client_id, data, num_examples])
    connection.close()

    cls.FEDERATED_DATA = sqlite_federated_data.SQLiteFederatedData.new(path)

  def test_metadata(self):
    federated_data = self.FEDERATED_DATA
    self.assertEqual(federated_data.num_clients(), 100)
    self.assertCountEqual(federated_data.client_ids(),
                          [f'{i:04x}'.encode('ascii') for i in range(100)])
    self.assertCountEqual(
        federated_data.client_sizes(),
        [(f'{i:04x}'.encode('ascii'), i + 1) for i in range(100)])
    self.assertEqual(federated_data.client_size(b'0000'), 1)
    with self.assertRaises(KeyError):
      federated_data.client_size(b'xxxx')

  def _assert_correct_client_dataset(self,
                                     client_id,
                                     client_dataset,
                                     feature='x',
                                     preprocessor=None):
    self.assertLen(client_id, 4)
    self.assertRegex(client_id, br'00[0-f][0-f]')
    i = int(client_id, base=16)
    self.assertCountEqual(client_dataset.raw_examples, [feature])
    npt.assert_array_equal(client_dataset.raw_examples[feature],
                           np.arange(i + 1))
    if preprocessor is None:
      self.assertIs(client_dataset.preprocessor,
                    self.FEDERATED_DATA._preprocess_batch)
    else:
      self.assertIs(client_dataset.preprocessor, preprocessor)

  def test_clients(self):
    client_ids = set()
    for client_id, client_dataset in self.FEDERATED_DATA.clients():
      self._assert_correct_client_dataset(client_id, client_dataset)
      client_ids.add(client_id)
    self.assertLen(client_ids, 100)

  def test_shuffled_clients(self):
    iterator = self.FEDERATED_DATA.shuffled_clients(buffer_size=10, seed=1)
    # Check client ids are shuffled.
    client_ids = []
    for client_id, client_dataset in itertools.islice(iterator, 10):
      self._assert_correct_client_dataset(client_id, client_dataset)
      client_ids.append(client_id)
    self.assertNotEqual(client_ids, [
        client_id
        for client_id, _ in itertools.islice(self.FEDERATED_DATA.clients(), 10)
    ])
    # Check iteration is repeated.
    for client_id, _ in itertools.islice(iterator, 190):
      client_ids.append(client_id)
    self.assertLen(client_ids, 200)
    self.assertNotEqual(client_ids[:100], client_ids[100:])

  def test_get_clients(self):
    client_ids = [b'0000', b'0010']
    for i, (client_id, client_dataset) in enumerate(
        self.FEDERATED_DATA.get_clients(client_ids)):
      self.assertEqual(client_id, client_ids[i])
      self._assert_correct_client_dataset(client_id, client_dataset)

    with self.assertRaises(KeyError):
      list(self.FEDERATED_DATA.get_clients([b'xxxx']))

  def test_get_client(self):
    self._assert_correct_client_dataset(b'0010',
                                        self.FEDERATED_DATA.get_client(b'0010'))
    with self.assertRaises(KeyError):
      self.FEDERATED_DATA.get_client(b'xxxx')

  def test_preprocess_client(self):
    federated_data = self.FEDERATED_DATA.preprocess_client(
        lambda _, x: {'y': x['x']})
    self.assertIs(federated_data._connection, self.FEDERATED_DATA._connection)
    self.assertIs(federated_data._parse_examples,
                  self.FEDERATED_DATA._parse_examples)
    self.assertIs(federated_data._start, self.FEDERATED_DATA._start)
    self.assertIs(federated_data._stop, self.FEDERATED_DATA._stop)
    self.assertIs(federated_data._preprocess_batch,
                  self.FEDERATED_DATA._preprocess_batch)
    self._assert_correct_client_dataset(
        *next(federated_data.clients()), feature='y')
    self._assert_correct_client_dataset(
        *next(federated_data.shuffled_clients(buffer_size=10, seed=1)),
        feature='y')
    self._assert_correct_client_dataset(
        *next(federated_data.get_clients([b'0000'])), feature='y')
    self._assert_correct_client_dataset(
        b'0010', federated_data.get_client(b'0010'), feature='y')

  def test_preprocess_batch(self):
    federated_data = self.FEDERATED_DATA.preprocess_batch(
        lambda x: {'z': x['x']})
    self.assertIs(federated_data._connection, self.FEDERATED_DATA._connection)
    self.assertIs(federated_data._parse_examples,
                  self.FEDERATED_DATA._parse_examples)
    self.assertIs(federated_data._start, self.FEDERATED_DATA._start)
    self.assertIs(federated_data._stop, self.FEDERATED_DATA._stop)
    self.assertIs(federated_data._preprocess_client,
                  self.FEDERATED_DATA._preprocess_client)
    self.assertCountEqual(
        federated_data._preprocess_batch({'x': np.arange(2)}), ['z'])
    self._assert_correct_client_dataset(
        *next(federated_data.clients()),
        preprocessor=federated_data._preprocess_batch)
    self._assert_correct_client_dataset(
        *next(federated_data.shuffled_clients(buffer_size=10, seed=1)),
        preprocessor=federated_data._preprocess_batch)
    self._assert_correct_client_dataset(
        *next(federated_data.get_clients([b'0000'])),
        preprocessor=federated_data._preprocess_batch)
    self._assert_correct_client_dataset(
        b'0010',
        federated_data.get_client(b'0010'),
        preprocessor=federated_data._preprocess_batch)

  def test_slice_metadata(self):
    federated_data = self.FEDERATED_DATA.slice(b'0001', b'0002')
    self.assertEqual(federated_data.num_clients(), 1)
    self.assertCountEqual(federated_data.client_ids(), [b'0001'])
    self.assertCountEqual(federated_data.client_sizes(), [(b'0001', 2)])
    self.assertEqual(federated_data.client_size(b'0001'), 2)
    for client_id in [b'0000', b'0002', b'xxxx']:
      with self.assertRaises(KeyError, msg=f'client_id: {client_id}'):
        federated_data.client_size(b'0000')

  def test_slice_iteration(self):
    federated_data = self.FEDERATED_DATA.slice(b'0001', b'0002')

    with self.subTest('clients'):
      clients = list(federated_data.clients())
      self.assertLen(clients, 1)
      self._assert_correct_client_dataset(*clients[0])

    with self.subTest('shuffled_clients'):
      shuffled_clients = list(
          itertools.islice(federated_data.shuffled_clients(buffer_size=10), 10))
      self.assertLen(shuffled_clients, 10)
      for client_id, client_dataset in shuffled_clients:
        self.assertEqual(client_id, b'0001')
        self._assert_correct_client_dataset(client_id, client_dataset)

    with self.subTest('get_clients'):
      clients = list(federated_data.get_clients([b'0001']))
      self.assertLen(clients, 1)
      self._assert_correct_client_dataset(*clients[0])
      with self.assertRaises(KeyError):
        list(federated_data.get_clients([b'0000']))

    with self.subTest('get_client'):
      self._assert_correct_client_dataset(b'0001',
                                          federated_data.get_client(b'0001'))
      with self.assertRaises(KeyError):
        federated_data.get_client(b'0000')

  def test_slice_monotonicity(self):
    # We can't slice a smaller federated data back into a big one.
    federated_data = self.FEDERATED_DATA.slice(b'0002', b'0004')
    client_ids = list(federated_data.client_ids())
    self.assertCountEqual(
        list(federated_data.slice(start=None, stop=b'9999').client_ids()),
        client_ids)
    self.assertCountEqual(
        list(federated_data.slice(start=b'').client_ids()), client_ids)
    self.assertCountEqual(
        list(federated_data.slice(start=b'', stop=b'9999').client_ids()),
        client_ids)


class SQLiteFederatedDataBuilderTest(SQLiteFederatedDataTest):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    path = os.path.join(FLAGS.test_tmpdir,
                        'test_sqlite_federated_data_builder.sqlite')

    # Create database file. First make sure the database file is empty.
    with open(path, 'w'):
      pass
    with sqlite_federated_data.SQLiteFederatedDataBuilder(path) as builder:
      client_ids_examples = [(f'{i:04x}'.encode('ascii'), {
          'x': np.arange(i + 1)
      }) for i in range(100)]
      builder.add_many(client_ids_examples)

    cls.FEDERATED_DATA = sqlite_federated_data.SQLiteFederatedData.new(path)


class TFFSQLiteClientsIteratorTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    path = os.path.join(FLAGS.test_tmpdir, 'test_tff_sqlite_reader.sqlite')
    split_name = 'train'

    # Create database file. First make sure the database file is empty.
    with open(path, 'w'):
      pass
    connection = sqlite3.connect(path)
    with connection:
      connection.execute("""
      CREATE TABLE examples (
          split_name TEXT NOT NULL,
          client_id TEXT NOT NULL,
          serialized_example_proto BLOB NOT NULL);""")

      connection.execute("""
      CREATE INDEX idx_examples_client_id
        ON examples (client_id);""")

      connection.execute("""
      CREATE INDEX idx_examples_client_id_split
        ON examples (split_name, client_id);""")

      connection.execute("""
      CREATE TABLE client_metadata (
        client_id TEXT NOT NULL,
        split_name TEXT NOT NULL,
        num_examples INTEGER NOT NULL);""")

      connection.execute("""
      CREATE INDEX idx_metadata_client_id
        ON client_metadata (client_id);""")

      for i in range(100):
        client_id = str(i)
        num_examples = i + 1
        for j in range(num_examples):
          tf_example = tf.train.Example(
              features=tf.train.Features(feature={
                  'x':
                      tf.train.Feature(
                          int64_list=tf.train.Int64List(value=[j]))
              }))
          serialized_example_proto = tf_example.SerializeToString()
          connection.execute('INSERT INTO examples VALUES (?, ?, ?);',
                             [split_name, client_id, serialized_example_proto])
        connection.execute('INSERT INTO client_metadata VALUES (?, ?, ?);',
                           [client_id, split_name, num_examples])
    connection.close()

    def parse_examples(vs):
      tf_examples = tf.io.parse_example(
          vs, features={'x': tf.io.FixedLenFeature(shape=(), dtype=tf.int64)})
      return tf.nest.map_structure(lambda t: t.numpy(), tf_examples)

    cls._path = path
    cls._parse_examples = parse_examples
    cls._split_name = split_name

  def test_iteration(self):
    client_ids = set()
    for client_id, client_dataset in sqlite_federated_data.TFFSQLiteClientsIterator(
        self._path, TFFSQLiteClientsIteratorTest._parse_examples,
        self._split_name):
      i = int(client_id)
      self.assertCountEqual(client_dataset.raw_examples, ['x'])
      npt.assert_array_equal(client_dataset.raw_examples['x'], np.arange(i + 1))
      client_ids.add(client_id)
    self.assertLen(client_ids, 100)


if __name__ == '__main__':
  absltest.main()
