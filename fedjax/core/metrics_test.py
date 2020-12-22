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
    self.assertAlmostEqual(loss, expected_loss)

  @parameterized.named_parameters(
      {
          'testcase_name': 'rank_1_no_reduce',
          'targets': [1, 0, 1],
          # (batch_size, num_classes).
          'preds': [[1.2, 0.4], [2.3, 0.1], [0.3, 3.2]],
          'mask_values': (0,),
          'reduce': False,
          'expected_loss': [1.1711007, 0., 0.05356275],
      },
      {
          'testcase_name': 'rank_1_reduce',
          'targets': [1, 0, 1],
          # (batch_size, num_classes).
          'preds': [[1.2, 0.4], [2.3, 0.1], [0.3, 3.2]],
          'mask_values': (0,),
          'reduce': True,
          'expected_loss': 0.612331725,
      },
      {
          'testcase_name': 'rank_2_no_reduce',
          'targets': [[1, 0], [0, 1], [0, 1]],
          # (batch_size, time_steps, num_classes).
          'preds': [[[2.3, 0.2], [1.2, 2.3]], [[0.4, 0.6], [0.1, 1.2]],
                    [[3.2, 2.1], [0.1, 2.0]]],
          'mask_values': (0,),
          'reduce': False,
          'expected_loss': [[2.2155194, 0.], [0., 0.28733534], [0., 0.1393867]],
      },
      {
          'testcase_name': 'rank_2_reduce',
          'targets': [[1, 0], [0, 1], [0, 1]],
          # (batch_size, time_steps, num_classes).
          'preds': [[[2.3, 0.2], [1.2, 2.3]], [[0.4, 0.6], [0.1, 1.2]],
                    [[3.2, 2.1], [0.1, 2.0]]],
          'mask_values': (0,),
          'reduce': True,
          'expected_loss': 0.880747,
      },
      {
          'testcase_name': 'rank_2_no_reduce_multi_mask',
          'targets': [[1, 0], [0, 2], [0, 1]],
          # (batch_size, time_steps, num_classes).
          'preds': [[[2.3, 0.2], [1.2, 2.3]], [[0.4, 0.6], [0.1, 1.2]],
                    [[3.2, 2.1], [0.1, 2.0]]],
          'mask_values': (0, 2),
          'reduce': False,
          'expected_loss': [[2.2155194, 0.], [0., 0.], [0., 0.1393867]],
      })
  def test_masked_cross_entropy_loss_fn(self, targets, preds, mask_values,
                                        reduce, expected_loss):
    targets, preds = jnp.array(targets), jnp.array(preds)
    loss = metrics.masked_cross_entropy_loss_fn(targets, preds, mask_values,
                                                reduce)
    self.assertAllClose(loss, expected_loss)

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
    self.assertAlmostEqual(accuracy, expected_accuracy)

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
    self.assertAlmostEqual(accuracy, expected_accuracy)

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
    self.assertAlmostEqual(accuracy, expected_accuracy)


if __name__ == '__main__':
  tf.test.main()
