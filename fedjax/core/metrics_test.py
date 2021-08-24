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
"""Tests for fedjax.core.metrics."""

from absl.testing import absltest
from absl.testing import parameterized

from fedjax.core import metrics

import jax.numpy as jnp
import numpy.testing as npt


class MeanStatTest(absltest.TestCase):

  def test_str(self):
    stat = metrics.MeanStat.new(2, 4)
    self.assertRegex(
        str(stat),
        r'MeanStat\(accum=DeviceArray\(2, .*\), weight=DeviceArray\(4, .*\)\) => 0.5'
    )

  def test_new(self):
    stat = metrics.MeanStat.new(jnp.array([2, 3, 1]), jnp.array([1, 0, 1]))
    npt.assert_array_equal(stat.accum, [2, 0, 1])
    npt.assert_array_equal(stat.weight, [1, 0, 1])

  def test_result(self):
    stat = metrics.MeanStat.new(2, 5)
    self.assertEqual(stat.result(), 0.4)

  def test_merge(self):
    stat_0 = metrics.MeanStat.new(1, 2)
    stat_1 = metrics.MeanStat.new(2, 3)
    merged_stat = stat_0.merge(stat_1)
    self.assertEqual(merged_stat.accum, 3)
    self.assertEqual(merged_stat.weight, 5)

  def test_reduce(self):
    stat = metrics.MeanStat.new(jnp.array([1, 2, 4]), jnp.array([1, 1, 0]))
    reduced_stat = stat.reduce()
    self.assertEqual(reduced_stat.accum, 3)
    self.assertEqual(reduced_stat.weight, 2)


class SumStatTest(absltest.TestCase):

  def test_str(self):
    stat = metrics.SumStat.new(2)
    self.assertRegex(str(stat), r'SumStat\(accum=DeviceArray\(2, .*\)\) => 2')

  def test_result(self):
    stat = metrics.SumStat.new(2)
    self.assertEqual(stat.result(), 2)

  def test_merge(self):
    stat_0 = metrics.SumStat.new(1)
    stat_1 = metrics.SumStat.new(2)
    merged_stat = stat_0.merge(stat_1)
    self.assertEqual(merged_stat.accum, 3)

  def test_reduce(self):
    stat = metrics.SumStat.new(jnp.array([1, 2, 1]))
    reduced_stat = stat.reduce()
    self.assertEqual(reduced_stat.accum, 4)


class MetricsTest(parameterized.TestCase):

  def test_cross_entropy_loss(self):
    example = {'y': jnp.array(1)}
    prediction = jnp.array([1.2, 0.4])
    metric = metrics.CrossEntropyLoss()
    with self.subTest('zero'):
      zero = metric.zero()
      self.assertEqual(zero.accum, 0)
      self.assertEqual(zero.weight, 0)
    with self.subTest('evaluate_example'):
      loss = metric.evaluate_example(example, prediction)
      self.assertAlmostEqual(loss.accum, 1.1711007)
      self.assertAlmostEqual(loss.weight, 1)

  @parameterized.named_parameters(
      {
          'testcase_name': 'correct',
          'target': 2,
          'prediction': [0, 0, 1],
          'expected_result': 1.,
      }, {
          'testcase_name': 'incorrect',
          'target': 1,
          'prediction': [1, 0, 0],
          'expected_result': 0.,
      })
  def test_accuracy(self, target, prediction, expected_result):
    example = {'y': jnp.array(target)}
    prediction = jnp.array(prediction)
    metric = metrics.Accuracy()
    with self.subTest('zero'):
      zero = metric.zero()
      self.assertEqual(zero.accum, 0)
      self.assertEqual(zero.weight, 0)
    with self.subTest('evaluate_example'):
      accuracy = metric.evaluate_example(example, prediction)
      self.assertEqual(accuracy.result(), expected_result)

  @parameterized.named_parameters(
      {
          'testcase_name': 'correct, 0 < k < classes',
          'kval': 3,
          'target': 2,
          'prediction': [0.7, 0., 0.2, -0.02, 0.6],
          'expected_result': 1.,
      }, {
          'testcase_name': 'incorrect, 0 < k < classes',
          'kval': 2,
          'target': 0,
          'prediction': [0., -0.2, 0.5, 0.55, 0.],
          'expected_result': 0.,
      }, {
          'testcase_name': 'incorrect, k = 0',
          'kval': 0,
          'target': 2,
          'prediction': [0., 0., 1., 0.],
          'expected_result': 0.,
      }, {
          'testcase_name': 'incorrect, k < 0',
          'kval': -5,
          'target': 2,
          'prediction': [0., 0., 1., 0.],
          'expected_result': 0.,
      }, {
          'testcase_name': 'correct, k > classes',
          'kval': 5,
          'target': 2,
          'prediction': [0., 0., 1., 0.],
          'expected_result': 1.,
      }, {
          'testcase_name': 'correct, same prediction for multiple classes',
          'kval': 3,
          'target': 0,
          'prediction': [0., .3, 1., 0.],
          'expected_result': 1.,
      }, {
          'testcase_name': 'incorrect, same prediction for multiple classes',
          'kval': 3,
          'target': 3,
          'prediction': [0., 0., 1., 0.],
          'expected_result': 0.,
      })
  def test_top_k_accuracy(self, kval, target, prediction, expected_result):
    example = {'y': jnp.array(target)}
    prediction = jnp.array(prediction)
    metric = metrics.TopKAccuracy(k=kval)
    with self.subTest('zero'):
      zero = metric.zero()
      self.assertEqual(zero.accum, 0)
      self.assertEqual(zero.weight, 0)
    with self.subTest('evaluate_example'):
      top_k_accuracy = metric.evaluate_example(example, prediction)
      self.assertEqual(top_k_accuracy.accum, expected_result)
      self.assertEqual(top_k_accuracy.weight, 1.)

  def test_sequence_token_cross_entropy_loss(self):
    example = {'y': jnp.array([1, 0, 1])}
    prediction = jnp.array([[1.2, 0.4], [2.3, 0.1], [0.3, 3.2]])
    metric = metrics.SequenceTokenCrossEntropyLoss()
    with self.subTest('zero'):
      zero = metric.zero()
      self.assertEqual(zero.accum, 0)
      self.assertEqual(zero.weight, 0)
    with self.subTest('evaluate_example'):
      loss = metric.evaluate_example(example, prediction)
      self.assertAlmostEqual(loss.accum, 1.2246635)
      self.assertAlmostEqual(loss.weight, 2)
    with self.subTest('per_position evaluate_example'):
      per_position_metric = metrics.SequenceTokenCrossEntropyLoss(
          per_position=True)
      per_position_loss = per_position_metric.evaluate_example(
          example, prediction)
      npt.assert_array_almost_equal(per_position_loss.accum,
                                    [1.1711007, 0., 0.05356275])
      npt.assert_array_equal(per_position_loss.weight, [1., 0., 1.])

  def test_sequence_cross_entropy_loss(self):
    example = {'y': jnp.array([1, 0, 1])}
    prediction = jnp.array([[1.2, 0.4], [2.3, 0.1], [0.3, 3.2]])
    metric = metrics.SequenceCrossEntropyLoss()
    with self.subTest('zero'):
      zero = metric.zero()
      self.assertEqual(zero.accum, 0)
      self.assertEqual(zero.weight, 0)
    with self.subTest('evaluate_example'):
      loss = metric.evaluate_example(example, prediction)
      self.assertAlmostEqual(loss.accum, 1.2246635)
      self.assertAlmostEqual(loss.weight, 1)

  def test_sequence_token_accuracy(self):
    example = {'y': jnp.array([1, 2, 2, 1, 3, 0])}
    # prediction = [1, 0, 2, 1, 3, 0].
    prediction = jnp.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0],
                            [0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
    logits_mask = (0., 0., 0., jnp.NINF)
    metric = metrics.SequenceTokenAccuracy(logits_mask=logits_mask)
    with self.subTest('zero'):
      zero = metric.zero()
      self.assertEqual(zero.accum, 0)
      self.assertEqual(zero.weight, 0)
    with self.subTest('evaluate_example'):
      accuracy = metric.evaluate_example(example, prediction)
      self.assertEqual(accuracy.accum, 3)
      self.assertEqual(accuracy.weight, 5)
    with self.subTest('per_position evaluate_example'):
      per_position_metric = metrics.SequenceTokenAccuracy(
          logits_mask=logits_mask, per_position=True)
      per_position_accuracy = per_position_metric.evaluate_example(
          example, prediction)
      npt.assert_array_almost_equal(per_position_accuracy.accum,
                                    [1., 0., 1., 1., 0., 0.])
      npt.assert_array_equal(per_position_accuracy.weight,
                             [1., 1., 1., 1., 1., 0.])

  def test_sequence_token_top_k_accuracy(self):
    example = {'y': jnp.array([1, 2, 2, 1, 3, 0])}
    prediction = jnp.array([[0, 1, 0.5, 0], [1, 0.5, 0, 0], [0.8, 0, 0.7, 0.2],
                            [0.5, 1, 0, 1], [0, 0.5, 0, 1], [0.5, 0, 0.9, 0.5]])
    logits_mask = (0., 0., 0., jnp.NINF)
    metric = metrics.SequenceTokenTopKAccuracy(k=2, logits_mask=logits_mask)
    with self.subTest('zero'):
      zero = metric.zero()
      self.assertEqual(zero.accum, 0)
      self.assertEqual(zero.weight, 0)
    with self.subTest('evaluate_example'):
      accuracy = metric.evaluate_example(example, prediction)
      self.assertEqual(accuracy.accum, 3)
      self.assertEqual(accuracy.weight, 5)
    with self.subTest('per_position evaluate_example'):
      per_position_metric = metrics.SequenceTokenTopKAccuracy(
          k=2, logits_mask=logits_mask, per_position=True)
      per_position_accuracy = per_position_metric.evaluate_example(
          example, prediction)
      npt.assert_array_almost_equal(per_position_accuracy.accum,
                                    [1., 0., 1., 1., 0., 0.])
      npt.assert_array_equal(per_position_accuracy.weight,
                             [1., 1., 1., 1., 1., 0.])

  def test_sequence_token_count(self):
    example = {'y': jnp.array([1, 2, 2, 3, 4, 0, 0])}
    prediction = jnp.array([])  # Unused.
    metric = metrics.SequenceTokenCount(masked_target_values=(0, 2))
    with self.subTest('zero'):
      zero = metric.zero()
      self.assertEqual(zero.accum, 0)
    with self.subTest('evaluate_example'):
      count = metric.evaluate_example(example, prediction)
      self.assertEqual(count.accum, 3)

  def test_sequence_count(self):
    example = {'y': jnp.array([1, 2, 2, 3, 4, 0, 0])}
    empty_example = {'y': jnp.array([0, 0, 0, 0, 0, 0, 0])}
    prediction = jnp.array([])  # Unused.
    metric = metrics.SequenceCount(masked_target_values=(0, 2))
    with self.subTest('zero'):
      zero = metric.zero()
      self.assertEqual(zero.accum, 0)
    with self.subTest('evaluate_example'):
      self.assertEqual(metric.evaluate_example(example, prediction).accum, 1)
      self.assertEqual(
          metric.evaluate_example(empty_example, prediction).accum, 0)

  @parameterized.named_parameters(
      {
          'testcase_name': 'untruncated',
          'target': [1, 2, 2, 3, 4, 0, 0],
          'expected_result': 0.,
      }, {
          'testcase_name': 'truncated',
          'target': [1, 2, 2, 3, 3, 3, 3],
          'expected_result': 1.,
      })
  def test_sequence_truncation_rate(self, target, expected_result):
    example = {'y': jnp.array(target)}
    prediction = jnp.array([])  # Unused.
    metric = metrics.SequenceTruncationRate(eos_target_value=4)
    with self.subTest('zero'):
      zero = metric.zero()
      self.assertEqual(zero.accum, 0)
      self.assertEqual(zero.weight, 0)
    with self.subTest('evaluate_example'):
      truncation_rate = metric.evaluate_example(example, prediction)
      self.assertEqual(truncation_rate.result(), expected_result)

  def test_sequence_token_oov_rate(self):
    example = {'y': jnp.array([1, 2, 2, 3, 4, 0, 0])}
    prediction = jnp.array([])  # Unused.
    metric = metrics.SequenceTokenOOVRate(oov_target_values=(2,))
    with self.subTest('zero'):
      zero = metric.zero()
      self.assertEqual(zero.accum, 0)
      self.assertEqual(zero.weight, 0)
    with self.subTest('evaluate_example'):
      oov_rate = metric.evaluate_example(example, prediction)
      self.assertEqual(oov_rate.accum, 2)
      self.assertEqual(oov_rate.weight, 5)
    with self.subTest('per_position evaluate_example'):
      per_position_metric = metrics.SequenceTokenOOVRate(
          oov_target_values=(2,), per_position=True)
      per_position_oov_rate = per_position_metric.evaluate_example(
          example, prediction)
      npt.assert_array_almost_equal(per_position_oov_rate.accum,
                                    [0., 1., 1., 0., 0., 0., 0.])
      npt.assert_array_equal(per_position_oov_rate.weight,
                             [1., 1., 1., 1., 1., 0., 0.])

  def test_sequence_length(self):
    example = {'y': jnp.array([1, 2, 3, 4, 0, 0])}
    prediction = jnp.array([])  # Unused.
    metric = metrics.SequenceLength()
    with self.subTest('zero'):
      zero = metric.zero()
      self.assertEqual(zero.accum, 0)
      self.assertEqual(zero.weight, 0)
    with self.subTest('evaluate_example'):
      sequence_length = metric.evaluate_example(example, prediction)
      self.assertEqual(sequence_length.accum, 4)
      self.assertEqual(sequence_length.weight, 1)

  def test_per_domain(self):
    metric = metrics.PerDomainMetric(metrics.Accuracy(), num_domains=4)

    stat = metric.zero()
    self.assertIsInstance(stat, metrics.MeanStat)
    npt.assert_array_equal(stat.accum, [0., 0., 0., 0.])
    npt.assert_array_equal(stat.weight, [0., 0., 0., 0.])

    stat = metric.evaluate_example({
        'y': jnp.array(1),
        'domain_id': 0
    }, jnp.array([0., 1.]))
    self.assertIsInstance(stat, metrics.MeanStat)
    npt.assert_array_equal(stat.accum, [1., 0., 0., 0.])
    npt.assert_array_equal(stat.weight, [1., 0., 0., 0.])

    stat = metric.evaluate_example({
        'y': jnp.array(0),
        'domain_id': 2
    }, jnp.array([0., 1.]))
    self.assertIsInstance(stat, metrics.MeanStat)
    npt.assert_array_equal(stat.accum, [0., 0., 0., 0.])
    npt.assert_array_equal(stat.weight, [0., 0., 1., 0.])

  @parameterized.named_parameters(
      {
          'testcase_name': 'correct',
          'target': 1,
          'prediction': [0., 1., 0.],
          'expected_result': [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
      }, {
          'testcase_name': 'incorrect',
          'target': 1,
          'prediction': [1., 0., 0.],
          'expected_result': [[0., 0., 0.], [1., 0., 0.], [0., 0., 0.]],
      })
  def test_confusion_matrix(self, target, prediction, expected_result):
    example = {'y': jnp.array(target)}
    prediction = jnp.array(prediction)
    num_classes = len(prediction)
    metric = metrics.ConfusionMatrix(num_classes=num_classes)
    with self.subTest('zero'):
      zero = metric.zero()
      npt.assert_array_equal(zero.accum, jnp.zeros((num_classes, num_classes)))
    with self.subTest('evaluate_example'):
      confusion_matrix = metric.evaluate_example(example, prediction)
      npt.assert_array_equal(confusion_matrix.result(), expected_result)

  def test_confusion_matrix_invalid_input(self):
    example = {'y': jnp.array(1)}
    prediction = jnp.array([0., 1.])
    metric = metrics.ConfusionMatrix(num_classes=3)
    with self.assertRaisesRegex(ValueError, 'Make sure num_classes is equal'):
      metric.evaluate_example(example, prediction)


if __name__ == '__main__':
  absltest.main()
