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
"""Tests for fedjax.experimental.metrics."""

from absl.testing import absltest
from absl.testing import parameterized

from fedjax.experimental import metrics

import jax.numpy as jnp
import numpy.testing as npt


class MeanStatTest(absltest.TestCase):

  def test_str(self):
    stat = metrics.MeanStat.new(2, 4)
    self.assertEqual(
        'MeanStat(accum=DeviceArray(2, dtype=int32), weight=DeviceArray(4, dtype=int32)) => 0.5',
        str(stat))

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
    self.assertEqual('SumStat(accum=DeviceArray(2, dtype=int32)) => 2',
                     str(stat))

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
    loss = metric.evaluate_example(example, prediction)
    self.assertAlmostEqual(loss.result(), 1.1711007)

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
    accuracy = metric.evaluate_example(example, prediction)
    self.assertEqual(accuracy.result(), expected_result)

  def test_sequence_token_cross_entropy_loss(self):
    example = {'y': jnp.array([1, 0, 1])}
    prediction = jnp.array([[1.2, 0.4], [2.3, 0.1], [0.3, 3.2]])
    metric = metrics.SequenceTokenCrossEntropyLoss()
    loss = metric.evaluate_example(example, prediction)
    self.assertAlmostEqual(loss.result(), 0.612331725)

  def test_sequence_cross_entropy_loss(self):
    example = {'y': jnp.array([1, 0, 1])}
    prediction = jnp.array([[1.2, 0.4], [2.3, 0.1], [0.3, 3.2]])
    metric = metrics.SequenceCrossEntropyLoss()
    loss = metric.evaluate_example(example, prediction)
    self.assertAlmostEqual(loss.result(), 1.2246635)

  def test_sequence_token_accuracy(self):
    example = {'y': jnp.array([1, 2, 2, 1, 0])}
    # prediction = [1, 0, 2, 1, 0].
    prediction = jnp.array([[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0],
                            [1, 0, 0]])
    metric = metrics.SequenceTokenAccuracy()
    accuracy = metric.evaluate_example(example, prediction)
    self.assertEqual(accuracy.result(), 0.75)  # 3 / 4.

  def test_sequence_token_count(self):
    example = {'y': jnp.array([1, 2, 2, 3, 4, 0, 0])}
    prediction = jnp.array([])  # Unused.
    metric = metrics.SequenceTokenCount(masked_target_values=(0, 2))
    count = metric.evaluate_example(example, prediction)
    self.assertEqual(count.result(), 3)

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
    truncation_rate = metric.evaluate_example(example, prediction)
    self.assertEqual(truncation_rate.result(), expected_result)

  def test_sequence_token_oov_rate(self):
    example = {'y': jnp.array([1, 2, 2, 3, 4, 0, 0])}
    prediction = jnp.array([])  # Unused.
    metric = metrics.SequenceTokenOOVRate(oov_target_values=(2,))
    oov_rate = metric.evaluate_example(example, prediction)
    self.assertEqual(oov_rate.result(), 0.4)  # 2 / 5.

  def test_sequence_length(self):
    example = {'y': jnp.array([1, 2, 3, 4, 0, 0])}
    prediction = jnp.array([])  # Unused.
    metric = metrics.SequenceLength()
    sequence_length = metric.evaluate_example(example, prediction)
    self.assertEqual(sequence_length.result(), 4.0)


if __name__ == '__main__':
  absltest.main()
