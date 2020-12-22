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
"""Tests for fedjax.training.federated_experiment."""

import collections
import os.path

from fedjax.core import test_util
from fedjax.training import federated_experiment
import tensorflow as tf


class FederatedExperimentTest(tf.test.TestCase):

  def test_run_federated_experiment(self):
    config = federated_experiment.FederatedExperimentConfig(
        root_dir=self.create_tempdir(),
        num_rounds=5,
        num_clients_per_round=2,
        checkpoint_frequency=3)
    num_examples = 3
    federated_algorithm = test_util.MockFederatedAlgorithm(
        num_examples=num_examples)

    state = federated_experiment.run_federated_experiment(
        config, federated_algorithm)

    self.assertEqual(
        state.count,
        config.num_rounds * config.num_clients_per_round * num_examples)
    self.assertTrue(
        os.path.exists(os.path.join(config.root_dir, 'checkpoint_00000003')))

  def test_run_federated_experiment_periodic_eval_fn_map(self):
    config = federated_experiment.FederatedExperimentConfig(
        root_dir=self.create_tempdir(),
        num_rounds=5,
        num_clients_per_round=2,
        eval_frequency=3)
    federated_algorithm = test_util.MockFederatedAlgorithm(num_examples=3)
    federated_data = federated_algorithm.federated_data
    model = federated_algorithm.model
    periodic_eval_fn_map = collections.OrderedDict(
        client_eval_1=federated_experiment.ClientEvaluationFn(
            federated_data, model, config),
        client_eval_2=federated_experiment.ClientEvaluationFn(
            federated_data, model, config),
        full_eval_1=federated_experiment.FullEvaluationFn(
            federated_data, model))

    federated_experiment.run_federated_experiment(
        config, federated_algorithm, periodic_eval_fn_map=periodic_eval_fn_map)

    self.assertTrue(
        os.path.exists(os.path.join(config.root_dir, 'client_eval_1')))
    self.assertTrue(
        os.path.exists(os.path.join(config.root_dir, 'client_eval_2')))
    self.assertTrue(
        os.path.exists(os.path.join(config.root_dir, 'full_eval_1')))

  def test_run_federated_experiment_final_eval_fn_map(self):
    config = federated_experiment.FederatedExperimentConfig(
        root_dir=self.create_tempdir(), num_rounds=5, num_clients_per_round=2)
    federated_algorithm = test_util.MockFederatedAlgorithm(num_examples=3)
    federated_data = federated_algorithm.federated_data
    model = federated_algorithm.model
    final_eval_fn_map = collections.OrderedDict(
        full_eval=federated_experiment.FullEvaluationFn(federated_data, model))

    federated_experiment.run_federated_experiment(
        config, federated_algorithm, final_eval_fn_map=final_eval_fn_map)

    self.assertTrue(
        os.path.exists(os.path.join(config.root_dir, 'full_eval.tsv')))


if __name__ == '__main__':
  tf.test.main()
