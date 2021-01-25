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
"""Tests for fedjax.models.toy_regression."""

from fedjax.datasets import toy_regression as toy_regression_data
from fedjax.models import toy_regression as toy_regression_model
import haiku as hk
import tensorflow as tf


class ToyRegressionTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._rng_seq = hk.PRNGSequence(42)
    self._batch_size = 3
    self._model = toy_regression_model.create_regression_model()
    data, _ = toy_regression_data.load_data(
        num_clients=10, num_domains=2, num_points=100, seed=10)
    self._batch = next(
        data.create_tf_dataset_for_client(data.client_ids[0]).repeat(
            self._batch_size).batch(self._batch_size).as_numpy_iterator())

  def test_backward_pass(self):
    params = self._model.init_params(rng=next(self._rng_seq))
    output = self._model.backward_pass(params, self._batch, next(self._rng_seq))
    self.assertGreaterEqual(output.loss, 0)
    self.assertEqual(output.num_examples, 1)

  def test_evaluate(self):
    params = self._model.init_params(rng=next(self._rng_seq))
    metrics = self._model.evaluate(params, self._batch)
    self.assertContainsSubset(['loss', 'num_examples'], metrics.keys())


if __name__ == '__main__':
  tf.test.main()
