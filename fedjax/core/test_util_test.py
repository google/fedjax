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
"""Tests for fedjax.core.test_util."""

from fedjax.core import test_util
import haiku as hk
import tensorflow as tf


class TestUtilsTest(tf.test.TestCase):

  def test_create_toy_data(self):
    data = test_util.create_toy_data(
        num_clients=10, num_clusters=2, num_classes=4, num_examples=5, seed=10)
    client_id = data.client_ids[3]
    client_data = list(data.create_tf_dataset_for_client(client_id))
    self.assertLen(data.client_ids, 10)
    self.assertLen(client_data, 5)
    self.assertAllEqual(client_data[0]['x'].shape, [1])
    self.assertAllEqual(client_data[0]['y'].shape, [])

  def test_create_toy_model(self):
    model = test_util.create_toy_model(
        num_classes=10, num_hidden_layers=2, feature_dim=5)
    params = model.init_params(next(hk.PRNGSequence(0)))
    self.assertEqual(list(params.keys()), ['linear', 'linear_1', 'linear_2'])
    self.assertEqual(hk.data_structures.tree_size(params), 750)

  def test_create_toy_example(self):
    data, model = test_util.create_toy_example(
        num_clients=10, num_clusters=2, num_classes=4, num_examples=5, seed=10)
    batch = next((data.create_tf_dataset_for_client(
        data.client_ids[0]).batch(3).as_numpy_iterator()))
    params = model.init_params(next(hk.PRNGSequence(0)))
    self.assertTupleEqual(model.apply_fn(params, None, batch).shape, (3, 4))

  def test_create_mock_state(self):
    state = test_util.create_mock_state(seed=0)
    self.assertEqual(list(state.params.keys()), ['linear'])
    self.assertEqual(hk.data_structures.tree_size(state.params), 124)

  def test_mock_federated_algorithm(self):
    num_clients = 10
    num_examples = 5
    num_rounds = 3
    algorithm = test_util.MockFederatedAlgorithm(
        num_clients=num_clients, num_examples=num_examples)

    state = algorithm.init_state()
    for _ in range(num_rounds):
      state = algorithm.run_round(state, algorithm.federated_data.client_ids)

    self.assertEqual(state.count, num_rounds * num_clients * num_examples)


if __name__ == '__main__':
  tf.test.main()
