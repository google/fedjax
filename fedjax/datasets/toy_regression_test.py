# Copyright 2020 Google LLC
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
"""Tests for fedjax.datasets.toy_regression."""

from fedjax.datasets import toy_regression
import tensorflow as tf


class ToyRegressionDataTest(tf.test.TestCase):

  def test_load_data(self):
    num_clients = 10
    train_data, test_data = toy_regression.load_data(
        num_clients=num_clients, num_domains=2, num_points=100, seed=10)
    client_id = train_data.client_ids[3]

    train_client_data = list(train_data.create_tf_dataset_for_client(client_id))
    test_client_data = list(test_data.create_tf_dataset_for_client(client_id))

    self.assertLen(train_data.client_ids, num_clients)
    self.assertEqual(train_data.client_ids, test_data.client_ids)
    self.assertNotAllEqual(train_client_data[0]['y'], test_client_data[0]['y'])


if __name__ == '__main__':
  tf.test.main()
