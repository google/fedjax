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

  @parameterized.named_parameters(
      {
          'testcase_name': 'rank_1',
          'targets': [1, 0, 1],
          # (batch_size, num_classes).
          'preds': [[1.2, 0.4], [2.3, 0.1], [0.3, 3.2]],
          'expected_loss': 0.44324896,
      },
      {
          'testcase_name': 'rank_2',
          'targets': [[1, 0], [0, 1], [0, 1]],
          # (batch_size, time_steps, num_classes).
          'preds': [[[2.3, 0.2], [1.2, 2.3]], [[0.4, 0.6], [0.1, 1.2]],
                    [[3.2, 2.1], [0.1, 2.0]]],
          'expected_loss': 0.8525085,
      })
  def test_cross_entropy_loss_fn(self, targets, preds, expected_loss):
    targets, preds = jnp.array(targets), jnp.array(preds)
    loss = metrics.cross_entropy_loss_fn(targets, preds)
    self.assertAlmostEqual(loss.result(), expected_loss)

  @parameterized.named_parameters(
      {
          'testcase_name': 'rank_1',
          'targets': [1, 0, 1],
          # (batch_size, num_classes).
          'preds': [[1.2, 0.4], [2.3, 0.1], [0.3, 3.2]],
          'mask_values': (0,),
          'expected_loss': 0.612331725,
      },
      {
          'testcase_name': 'rank_2',
          'targets': [[1, 0], [0, 1], [0, 1]],
          # (batch_size, time_steps, num_classes).
          'preds': [[[2.3, 0.2], [1.2, 2.3]], [[0.4, 0.6], [0.1, 1.2]],
                    [[3.2, 2.1], [0.1, 2.0]]],
          'mask_values': (0,),
          'expected_loss': 0.880747,
      },
      {
          'testcase_name': 'rank_2_multi_mask',
          'targets': [[1, 0], [0, 2], [0, 1]],
          # (batch_size, time_steps, num_classes).
          'preds': [[[2.3, 0.2], [1.2, 2.3]], [[0.4, 0.6], [0.1, 1.2]],
                    [[3.2, 2.1], [0.1, 2.0]]],
          'mask_values': (0, 2),
          'expected_loss': 1.17745305,
      })
  def test_masked_cross_entropy_loss_fn(self, targets, preds, mask_values,
                                        expected_loss):
    targets, preds = jnp.array(targets), jnp.array(preds)
    loss = metrics.masked_cross_entropy_loss_fn(targets, preds, mask_values)
    self.assertAllClose(loss.result(), expected_loss)

  @parameterized.named_parameters(
      {
          'testcase_name': 'rank_1',
          'targets': [1, 0, 2, 0],
          'preds': [[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]],
          'expected_accuracy': 3. / 4.,
      }, {
          'testcase_name': 'rank_2',
          'targets': [[1, 0], [1, 1], [1, 1], [0, 2]],
          'preds': [[[0, 1, 0], [1, 0, 0]], [[0, 0, 1], [0, 0, 1]],
                    [[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 1, 0]]],
          'expected_accuracy': 4. / 8.,
      })
  def test_accuracy_fn(self, targets, preds, expected_accuracy):
    targets, preds = jnp.array(targets), jnp.array(preds)
    accuracy = metrics.accuracy_fn(targets, preds)
    self.assertAlmostEqual(accuracy.result(), expected_accuracy)

  @parameterized.named_parameters(
      {
          'testcase_name': 'rank_1',
          'targets': [1, 0, 2, 0],
          'preds': [[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]],
          'mask_values': (0,),
          'expected_accuracy': 2. / 2.,
      }, {
          'testcase_name': 'rank_2',
          'targets': [[1, 0], [1, 1], [1, 1], [0, 2]],
          'preds': [[[0, 1, 0], [1, 0, 0]], [[0, 0, 1], [0, 0, 1]],
                    [[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 1, 0]]],
          'mask_values': (0,),
          'expected_accuracy': 2. / 6.,
      }, {
          'testcase_name': 'rank_2_multi_mask',
          'targets': [[1, 0], [1, 1], [2, 1], [0, 2]],
          'preds': [[[0, 1, 0], [1, 0, 0]], [[0, 0, 1], [0, 0, 1]],
                    [[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 1, 0]]],
          'mask_values': (0, 2),
          'expected_accuracy': 2. / 4.,
      })
  def test_masked_accuracy_fn(self, targets, preds, mask_values,
                              expected_accuracy):
    targets, preds = jnp.array(targets), jnp.array(preds)
    accuracy = metrics.masked_accuracy_fn(targets, preds, mask_values)
    self.assertAlmostEqual(accuracy.result(), expected_accuracy)

  @parameterized.named_parameters(
      {
          'testcase_name': 'rank_1',
          'targets': [1, 0, 2, 0],
          'preds': [[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]],
          'mask_values': (0,),
          'logits_mask': [0, jnp.NINF, 0],
          'expected_accuracy': 1. / 2.,
      }, {
          'testcase_name': 'rank_2',
          'targets': [[1, 0], [1, 1], [1, 1], [0, 2]],
          'preds': [[[0, 1, 0], [1, 0, 0]], [[0, 0, 1], [0, 0, 1]],
                    [[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 1, 0]]],
          'mask_values': (0,),
          'logits_mask': [0, jnp.NINF, 0],
          'expected_accuracy': 0. / 6.,
      }, {
          'testcase_name': 'rank_2_multi_mask',
          'targets': [[1, 0], [1, 1], [2, 1], [0, 2]],
          'preds': [[[0, 1, 0], [1, 0, 0]], [[0, 0, 1], [0, 0, 1]],
                    [[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 1, 0]]],
          'mask_values': (0, 2),
          'logits_mask': [0, jnp.NINF, 0],
          'expected_accuracy': 0. / 4.,
      })
  def test_masked_accuracy_fn_with_logits_mask(self, targets, preds,
                                               mask_values, logits_mask,
                                               expected_accuracy):
    targets, preds, logits_mask = jnp.array(targets), jnp.array(
        preds), jnp.array(logits_mask)
    accuracy = metrics.masked_accuracy_fn_with_logits_mask(
        targets, preds, logits_mask, mask_values)
    self.assertAlmostEqual(accuracy.result(), expected_accuracy)

  def test_str_version_of_metric(self):
    metric = metrics.MeanMetric(total=2, count=4)
    self.assertEqual('MeanMetric(total=2, count=4) => 0.5', str(metric))

  def test_raises_on_non_scalar_value(self):
    with self.assertRaises(TypeError):
      metrics.MeanMetric(total=jnp.array([1]), count=jnp.array([2]))
    with self.assertRaises(TypeError):
      metrics.CountMetric(count=jnp.array([3]))

  def test_masked_count(self):
    count = metrics.masked_count(
        targets=jnp.array([[1, 3], [1, 1], [2, 1], [0, 2]]), mask_values=(0, 2))
    self.assertEqual(count.result(), 5)

  def test_truncation_rate(self):
    targets = jnp.array([[1, 0], [1, 3], [2, 1], [2, 0], [0, 0]])
    eos_value = 3
    pad_value = 0
    truncation_rate = metrics.truncation_rate(targets, eos_value, pad_value)
    self.assertEqual(truncation_rate.result(), 0.75)  # 3 / 4.

  def test_oov_rate(self):
    targets = jnp.array([[1, 0], [1, 3], [3, 1], [0, 2]])
    oov_values = (3,)
    mask_values = (0, 2)
    oov_rate = metrics.oov_rate(targets, oov_values, mask_values)
    self.assertEqual(oov_rate.result(), 0.4)  # 2 / 5.

  def test_sequence_length(self):
    targets = jnp.array([[1, 0], [1, 3], [2, 1], [2, 0], [0, 0]])
    pad_value = 0
    sequence_length = metrics.sequence_length(targets, pad_value)
    self.assertEqual(sequence_length.result(), 1.5)  # 6 / 4.


if __name__ == '__main__':
  tf.test.main()
