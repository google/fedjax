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
"""Tests for fedjax.models.emnist."""

from fedjax.datasets import emnist as emnist_data
from fedjax.models import emnist as emnist_model
import haiku as hk
import numpy as np
import tensorflow as tf


def _mock_emnist_data(only_digits=True, seed=0):
  """Returns fixed example to avoid reading data over network in test."""
  num_classes = 10 if only_digits else 62
  dataset = tf.data.Dataset.from_tensor_slices({
      'pixels': np.random.RandomState(seed).random_sample((1, 28, 28)),
      'label': [np.random.RandomState(seed).randint(num_classes)],
  })
  return emnist_data.flip_and_expand(dataset)


class EmnistHaikuDenseTest(tf.test.TestCase):

  def create_model(self):
    return emnist_model.create_dense_model(only_digits=False)

  def setUp(self):
    super().setUp()
    self._rng_seq = hk.PRNGSequence(42)
    self._batch_size = 4
    self._batch = next(
        _mock_emnist_data(only_digits=False).repeat(self._batch_size).batch(
            self._batch_size).as_numpy_iterator())
    self._model = self.create_model()

  def test_backward_pass(self):
    params = self._model.init_params(rng=next(self._rng_seq))
    output = self._model.backward_pass(params, self._batch, next(self._rng_seq))
    self.assertGreaterEqual(output.loss, 0)
    self.assertEqual(output.num_examples, self._batch_size)

  def test_evaluate(self):
    params = self._model.init_params(rng=next(self._rng_seq))
    metrics = self._model.evaluate(params, self._batch)
    self.assertContainsSubset(['loss', 'accuracy', 'num_examples'],
                              metrics.keys())


class EmnistHaikuLogisticTest(EmnistHaikuDenseTest):

  def create_model(self):
    return emnist_model.create_logistic_model(only_digits=False)


class EmnistHaikuConvTest(EmnistHaikuDenseTest):

  def create_model(self):
    return emnist_model.create_conv_model(only_digits=False)


class EmnistStaxDenseTest(EmnistHaikuDenseTest):

  def create_model(self):
    return emnist_model.create_stax_dense_model(only_digits=False)


if __name__ == '__main__':
  tf.test.main()
