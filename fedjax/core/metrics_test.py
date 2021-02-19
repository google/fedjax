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
"""Tests for fedjax.core.metrics."""

from absl.testing import parameterized
from fedjax.core import metrics
import jax.numpy as jnp
import tensorflow as tf


class MetricsTest(tf.test.TestCase, parameterized.TestCase):

  def test_create_mask(self):
    values = jnp.array([[0, 1], [2, 1], [3, 1]])
    mask_values = (1, 2)
    mask = metrics.create_mask(values, mask_values)
    self.assertAllEqual(mask, [[True, False], [False, False], [True, False]])

  @parameterized.named_parameters(
      {
          'testcase_name': 'rank_1',
          'shape': (3,),
          'batch_mask': [True, True, False],
          'expected_mask': [True, True, False],
      }, {
          'testcase_name':
              'rank_3',
          'shape': (3, 2, 2),
          'batch_mask': [True, True, False],
          'expected_mask': [[[True, True], [True, True]],
                            [[True, True], [True, True]],
                            [[False, False], [False, False]]],
      })
  def test_broadcast_batch_mask(self, shape, batch_mask, expected_mask):
    values = jnp.zeros(shape)
    batch_mask = jnp.array(batch_mask)
    mask = metrics.broadcast_batch_mask(values, batch_mask)
    self.assertAllEqual(mask, expected_mask)

  @parameterized.named_parameters(
      {
          'testcase_name': 'rank_1',
          'targets': [1, 0, 1],
          # (batch_size, num_classes).
          'preds': [[1.2, 0.4], [2.3, 0.1], [0.3, 3.2]],
          'targets_mask': [True, True, False],
          'expected_loss': 0.63809204,
      },
      {
          'testcase_name': 'rank_2',
          'targets': [[1, 0], [0, 2], [0, 1]],
          # (batch_size, time_steps, num_classes).
          'preds': [[[2.3, 0.2], [1.2, 2.3]], [[0.4, 0.6], [0.1, 1.2]],
                    [[3.2, 2.1], [0.1, 2.0]]],
          'targets_mask': [[True, False], [False, False], [False, True]],
          'expected_loss': 1.17745305,
      })
  def test_cross_entropy_loss_fn(self, targets, preds, targets_mask,
                                 expected_loss):
    loss = metrics.cross_entropy_loss_fn(
        jnp.array(targets), jnp.array(preds), jnp.array(targets_mask))
    self.assertAlmostEqual(loss.result(), expected_loss)

  @parameterized.named_parameters(
      {
          'testcase_name': 'rank_1',
          'targets': [1, 0, 2, 0, 1],
          'preds': [[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0]],
          'targets_mask': [True, True, False, True, True],
          'expected_accuracy': 3. / 4.,
      }, {
          'testcase_name': 'rank_2',
          'targets': [[1, 0], [1, 1], [1, 1], [0, 2]],
          'preds': [[[0, 1, 0], [1, 0, 0]], [[0, 0, 1], [0, 0, 1]],
                    [[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 1, 0]]],
          'targets_mask': [[True, False], [True, True], [True, True],
                           [False, False]],
          'expected_accuracy': 2. / 5.,
      })
  def test_accuracy_fn(self, targets, preds, targets_mask, expected_accuracy):
    accuracy = metrics.accuracy_fn(
        jnp.array(targets), jnp.array(preds), jnp.array(targets_mask))
    self.assertAlmostEqual(accuracy.result(), expected_accuracy)

  @parameterized.named_parameters(
      {
          'testcase_name': 'rank_1',
          'targets': [1, 0, 2, 0],
          'preds': [[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]],
          'targets_mask': [True, False, True, False],
          'logits_mask': [0, jnp.NINF, 0],
          'expected_accuracy': 1. / 2.,
      }, {
          'testcase_name': 'rank_2',
          'targets': [[1, 0], [1, 1], [2, 1], [0, 2]],
          'preds': [[[0, 1, 0], [1, 0, 0]], [[0, 0, 1], [0, 0, 1]],
                    [[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 1, 0]]],
          'targets_mask': [[True, False], [True, True], [False, True],
                           [False, False]],
          'logits_mask': [0, jnp.NINF, 0],
          'expected_accuracy': 0. / 4.,
      })
  def test_accuracy_fn_with_logits_mask(self, targets, preds, targets_mask,
                                        logits_mask, expected_accuracy):
    accuracy = metrics.accuracy_fn_with_logits_mask(
        jnp.array(targets), jnp.array(preds), jnp.array(targets_mask),
        jnp.array(logits_mask))
    self.assertAlmostEqual(accuracy.result(), expected_accuracy)

  def test_str_version_of_metric(self):
    metric = metrics.MeanMetric(total=2, count=4)
    self.assertEqual('MeanMetric(total=2, count=4) => 0.5', str(metric))

  def test_raises_on_non_scalar_value(self):
    with self.assertRaises(TypeError):
      metrics.MeanMetric(total=jnp.array([1]), count=jnp.array([2]))
    with self.assertRaises(TypeError):
      metrics.CountMetric(count=jnp.array([3]))

  def test_count(self):
    count = metrics.count(
        jnp.array([[True, False], [False, False], [True, False]]))
    self.assertEqual(count.result(), 2)

  def test_truncation_rate(self):
    targets = jnp.array([[1, 0], [1, 3], [2, 1], [2, 0], [0, 0]])
    targets_mask = jnp.array([[True, True], [True, True], [True, True],
                              [True, False], [False, False]])
    eos_value = 3
    pad_value = 0
    truncation_rate = metrics.truncation_rate(targets, targets_mask, eos_value,
                                              pad_value)
    self.assertEqual(truncation_rate.result(), 0.75)  # 3 / 4.

  def test_oov_rate(self):
    targets = jnp.array([[1, 0], [1, 3], [3, 1], [0, 2]])
    targets_mask = jnp.array([[True, True], [True, True], [True, False],
                              [False, False]])
    oov_values = (3,)
    oov_rate = metrics.oov_rate(targets, targets_mask, oov_values)
    self.assertEqual(oov_rate.result(), 0.4)  # 2 / 5.

  def test_sequence_length(self):
    targets = jnp.array([[1, 0], [1, 3], [2, 1], [2, 0], [0, 0]])
    targets_mask = jnp.array([[True, True], [True, True], [True, True],
                              [True, True], [False, False]])
    pad_value = 0
    sequence_length = metrics.sequence_length(targets, targets_mask, pad_value)
    self.assertEqual(sequence_length.result(), 1.5)  # 6 / 4.


if __name__ == '__main__':
  tf.test.main()
