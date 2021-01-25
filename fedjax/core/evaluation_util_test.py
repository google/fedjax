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
"""Tests for fedjax.core.evaluation_util."""

import collections

from absl.testing import parameterized
from fedjax.core import dataset_util
from fedjax.core import evaluation_util
from fedjax.core import metrics
from fedjax.core import test_util
import haiku as hk
import tensorflow as tf


class EvaluationUtilTest(tf.test.TestCase, parameterized.TestCase):

  def test_evaluate_single_client(self):
    num_clients = 10
    num_classes = 10
    num_examples = 100
    client_data_hparams = dataset_util.ClientDataHParams(
        batch_size=20, num_epochs=3)
    data, model = test_util.create_toy_example(
        num_clients=num_clients,
        num_clusters=4,
        num_classes=num_classes,
        num_examples=num_examples,
        seed=0)
    rng_seq = hk.PRNGSequence(0)
    init_params = model.init_params(next(rng_seq))
    dataset = dataset_util.preprocess_tf_dataset(
        dataset_util.create_tf_dataset_for_clients(data), client_data_hparams)

    with self.subTest('tf dataset'):
      init_metrics = evaluation_util.evaluate_single_client(
          dataset=dataset, model=model, params=init_params)
      self.assertLess(0.0, init_metrics['loss'])

    with self.subTest('plain iterator'):
      init_metrics = evaluation_util.evaluate_single_client(
          dataset=dataset.as_numpy_iterator(), model=model, params=init_params)
      self.assertLess(0.0, init_metrics['loss'])

  def test_aggregate_metrics(self):
    metrics_dict = [
        collections.OrderedDict(
            loss=metrics.MeanMetric(total=2, count=11),
            num_examples=metrics.CountMetric(count=3)),
        collections.OrderedDict(
            loss=metrics.MeanMetric(total=6, count=9),
            num_examples=metrics.CountMetric(count=7))
    ]

    aggregated = evaluation_util.aggregate_metrics(metrics_dict)

    self.assertAlmostEqual(aggregated['loss'], 0.4)
    self.assertEqual(aggregated['num_examples'], 10.)

  def test_aggregate_metrics_empty(self):
    aggregated = evaluation_util.aggregate_metrics([])

    self.assertEmpty(aggregated)


if __name__ == '__main__':
  tf.test.main()
