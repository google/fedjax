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
"""Tests for in_memory_federated_data."""

from absl.testing import absltest
from fedjax.core import client_datasets
from fedjax.core import in_memory_federated_data
import numpy as np
import numpy.testing as npt


class InMemoryFederatedDataTest(absltest.TestCase):

  def test_in_memory_data_test(self):
    client_a_data = {
        'x': np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        'y': np.array([7, 8])
    }
    client_b_data = {'x': np.array([[9.0, 10.0, 11.0]]), 'y': np.array([12])}
    client_to_data_mapping = {'a': client_a_data, 'b': client_b_data}

    federated_data = in_memory_federated_data.InMemoryFederatedData(
        client_to_data_mapping)

    with self.subTest('num_clients'):
      self.assertEqual(federated_data.num_clients(), 2)

    with self.subTest('client_ids'):
      self.assertEqual(list(federated_data.client_ids()), ['a', 'b'])

    with self.subTest('client_sizes'):
      self.assertEqual(
          list(federated_data.client_sizes()), [('a', 2), ('b', 1)])

    with self.subTest('client_size'):
      self.assertEqual(federated_data.client_size('a'), 2)

    with self.subTest('get_client'):
      self.assertEqual(
          federated_data.get_client('a').all_examples(), client_a_data)

    with self.subTest('get_clients'):
      clients_data = list(federated_data.get_clients(['a']))
      self.assertLen(clients_data, 1)
      client_id, client = clients_data[0]
      self.assertEqual(client_id, 'a')
      self.assertEqual(client.all_examples(), client_a_data)

    with self.subTest('clients'):
      all_client_datasets = list(federated_data.clients())
      self.assertEqual(
          [(key, value.all_examples()) for (key, value) in all_client_datasets],
          [(key, value) for key, value in client_to_data_mapping.items()])

    with self.subTest('preprocess_client'):

      def preprocess_client(client_id, examples):
        del client_id
        examples['x'] = 2 * examples['x']
        return examples

      new_federeated_data = federated_data.preprocess_client(preprocess_client)
      new_client_a_data = {
          'x': np.array([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]]),
          'y': np.array([7, 8])
      }
      npt.assert_array_equal(
          new_federeated_data.get_client('a').all_examples()['x'],
          new_client_a_data['x'])
      npt.assert_array_equal(
          new_federeated_data.get_client('a').all_examples()['y'],
          new_client_a_data['y'])

    with self.subTest('preprocess_batch'):

      def preprocess_batch(examples):
        examples['x'] = 2 * examples['x']
        return examples

      new_federeated_data = federated_data.preprocess_batch(preprocess_batch)
      new_client_a_data = {
          'x': np.array([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]]),
          'y': np.array([7, 8])
      }
      batched_client_a_data = list(
          new_federeated_data.get_client('a').batch(
              hparams=client_datasets.BatchHParams(2)))[0]
      npt.assert_array_equal(batched_client_a_data['x'], new_client_a_data['x'])
      npt.assert_array_equal(batched_client_a_data['y'], new_client_a_data['y'])


if __name__ == '__main__':
  absltest.main()
