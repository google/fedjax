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
"""Tests for fedjax.core.federated_data."""

import itertools
import os.path
import zlib

from absl import flags
from absl.testing import absltest
from fedjax.core import federated_data
from fedjax.core import serialization
from fedjax.core import sqlite_federated_data
import numpy as np
import numpy.testing as npt
import sqlite3

FLAGS = flags.FLAGS


class RepeatableIteratorTest(absltest.TestCase):

  def test_two_passes(self):
    # range(5) is repeatable, iter(range(5)) is not.
    it = federated_data.RepeatableIterator(iter(range(5)))
    self.assertListEqual(list(it), [0, 1, 2, 3, 4])
    self.assertListEqual(list(it), [0, 1, 2, 3, 4])

  def test_no_copy_cases(self):
    for values in [[1, 2, 3], (1, 2, 3), '123', b'123', {'1': 2, '3': 4}]:
      it = federated_data.RepeatableIterator(values)
      self.assertIs(values, it._buf)
      self.assertListEqual(list(it), list(values))
      self.assertListEqual(list(it), list(values))


class HelperTest(absltest.TestCase):
  """Tests for helper functions."""

  def test_intersect_slice_ranges(self):
    with self.subTest('start'):
      self.assertEqual(
          federated_data.intersect_slice_ranges(None, None, b'x', None),
          (b'x', None))
      self.assertEqual(
          federated_data.intersect_slice_ranges(b'x', None, None, None),
          (b'x', None))
      self.assertEqual(
          federated_data.intersect_slice_ranges(b'a', None, b'x', None),
          (b'x', None))
      self.assertEqual(
          federated_data.intersect_slice_ranges(b'x', None, b'a', None),
          (b'x', None))

    with self.subTest('stop'):
      self.assertEqual(
          federated_data.intersect_slice_ranges(None, None, None, b'x'),
          (None, b'x'))
      self.assertEqual(
          federated_data.intersect_slice_ranges(None, b'x', None, None),
          (None, b'x'))
      self.assertEqual(
          federated_data.intersect_slice_ranges(None, b'a', None, b'x'),
          (None, b'a'))
      self.assertEqual(
          federated_data.intersect_slice_ranges(None, b'x', None, b'a'),
          (None, b'a'))

    with self.subTest('start and stop'):
      self.assertEqual(
          federated_data.intersect_slice_ranges(None, None, None, None),
          (None, None))
      self.assertEqual(
          federated_data.intersect_slice_ranges(b'a', b'g', b'c', b'h'),
          (b'c', b'g'))


class SubsetFederatedDataTest(absltest.TestCase):

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

    cls.BASE_FEDERATED_DATA = sqlite_federated_data.SQLiteFederatedData.new(
        path)
    cls.CLIENT_IDS = [b'0000', b'000f', b'0012']
    cls.FEDERATED_DATA = federated_data.SubsetFederatedData(
        cls.BASE_FEDERATED_DATA, cls.CLIENT_IDS)

  def test_metadata(self):
    self.assertEqual(self.FEDERATED_DATA.num_clients(), 3)
    self.assertEqual(
        list(self.FEDERATED_DATA.client_ids()), [b'0000', b'000f', b'0012'])
    self.assertEqual(
        list(self.FEDERATED_DATA.client_sizes()), [(b'0000', 1), (b'000f', 16),
                                                   (b'0012', 19)])
    for client_id, size in [(b'0000', 1), (b'000f', 16), (b'0012', 19)]:
      self.assertEqual(self.FEDERATED_DATA.client_size(client_id), size)
    for client_id in [b'xxxx', b'0001', b'1111']:
      with self.assertRaises(KeyError):
        self.FEDERATED_DATA.client_size(client_id)

  def _assert_correct_client_dataset(self,
                                     client_id,
                                     client_dataset,
                                     feature='x',
                                     preprocessor=None):
    self.assertLen(client_id, 4)
    self.assertRegex(client_id, br'00[0-f][0-f]')
    self.assertIn(client_id, self.CLIENT_IDS)
    i = int(client_id, base=16)
    self.assertCountEqual(client_dataset.raw_examples, [feature])
    npt.assert_array_equal(client_dataset.raw_examples[feature],
                           np.arange(i + 1))
    if preprocessor is None:
      self.assertIs(client_dataset.preprocessor,
                    self.BASE_FEDERATED_DATA._preprocess_batch)
    else:
      self.assertIs(client_dataset.preprocessor, preprocessor)

  def test_clients(self):
    client_ids = []
    for client_id, dataset in self.FEDERATED_DATA.clients():
      self.assertIn(client_id, self.CLIENT_IDS)
      self._assert_correct_client_dataset(client_id, dataset)
      client_ids.append(client_id)
    self.assertCountEqual(client_ids, self.CLIENT_IDS)

  def test_shuffled_clients(self):
    client_ids = []
    for client_id, dataset in itertools.islice(
        self.FEDERATED_DATA.shuffled_clients(buffer_size=3, seed=1), 5):
      self.assertIn(client_id, self.CLIENT_IDS)
      self._assert_correct_client_dataset(client_id, dataset)
      client_ids.append(client_id)
    self.assertListEqual(client_ids,
                         [b'0000', b'0012', b'000f', b'000f', b'0012'])

  def test_get_clients(self):
    client_ids = []
    for client_id, dataset in self.FEDERATED_DATA.get_clients(
        [b'0000', b'0012']):
      self.assertIn(client_id, self.CLIENT_IDS)
      self._assert_correct_client_dataset(client_id, dataset)
      client_ids.append(client_id)
    self.assertListEqual(client_ids, [b'0000', b'0012'])

    for client_id in [b'xxxx', b'0001', b'1111']:
      with self.assertRaises(KeyError):
        list(self.FEDERATED_DATA.get_clients([client_id]))

  def test_get_client(self):
    for client_id in self.CLIENT_IDS:
      dataset = self.FEDERATED_DATA.get_client(client_id)
      self._assert_correct_client_dataset(client_id, dataset)

    for client_id in [b'xxxx', b'0001', b'1111']:
      with self.assertRaises(KeyError):
        list(self.FEDERATED_DATA.get_client(client_id))

  def test_slice(self):
    self.assertCountEqual(
        list(self.FEDERATED_DATA.slice().client_ids()),
        [b'0000', b'000f', b'0012'])
    self.assertCountEqual(
        list(self.FEDERATED_DATA.slice(start=b'000f').client_ids()),
        [b'000f', b'0012'])
    self.assertCountEqual(
        list(self.FEDERATED_DATA.slice(stop=b'000f').client_ids()), [b'0000'])
    self.assertCountEqual(
        list(
            self.FEDERATED_DATA.slice(start=b'000f',
                                      stop=b'0012').client_ids()), [b'000f'])

  def test_preprocess_client(self):
    fd = self.FEDERATED_DATA.preprocess_client(lambda _, x: {'y': x['x']})
    self.assertIs(fd._client_ids, self.FEDERATED_DATA._client_ids)
    self.assertIs(fd._base._connection, self.BASE_FEDERATED_DATA._connection)
    self.assertIs(fd._base._preprocess_batch,
                  self.BASE_FEDERATED_DATA._preprocess_batch)
    self._assert_correct_client_dataset(*next(fd.clients()), feature='y')
    self._assert_correct_client_dataset(
        *next(fd.shuffled_clients(buffer_size=10, seed=1)), feature='y')
    self._assert_correct_client_dataset(
        *next(fd.get_clients([b'0000'])), feature='y')
    self._assert_correct_client_dataset(
        b'0012', fd.get_client(b'0012'), feature='y')

  def test_preprocess_batch(self):
    fd = self.FEDERATED_DATA.preprocess_batch(lambda x: {'z': x['x']})
    self.assertIs(fd._client_ids, self.FEDERATED_DATA._client_ids)
    self.assertIs(fd._base._connection, self.BASE_FEDERATED_DATA._connection)
    self.assertIs(fd._base._preprocess_client,
                  self.BASE_FEDERATED_DATA._preprocess_client)
    self.assertCountEqual(
        fd._base._preprocess_batch({'x': np.arange(2)}), ['z'])
    self._assert_correct_client_dataset(
        *next(fd.clients()), preprocessor=fd._base._preprocess_batch)
    self._assert_correct_client_dataset(
        *next(fd.shuffled_clients(buffer_size=10, seed=1)),
        preprocessor=fd._base._preprocess_batch)
    self._assert_correct_client_dataset(
        *next(fd.get_clients([b'0000'])),
        preprocessor=fd._base._preprocess_batch)
    self._assert_correct_client_dataset(
        b'0012',
        fd.get_client(b'0012'),
        preprocessor=fd._base._preprocess_batch)

  def test_bad_client_ids(self):
    with self.assertRaisesRegex(
        ValueError, 'Some client ids are not in the base FederatedData'):
      federated_data.SubsetFederatedData(self.BASE_FEDERATED_DATA, [b'xxxx'])


if __name__ == '__main__':
  absltest.main()
