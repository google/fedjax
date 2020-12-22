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
"""Tests for fedjax.algorithms.agnostic_fed_avg."""

from absl.testing import parameterized
from fedjax import core
from fedjax.algorithms import agnostic_fed_avg
from fedjax.datasets import toy_regression as toy_regression_data
from fedjax.models import toy_regression as toy_regression_model
import jax.numpy as jnp
import tensorflow as tf


class AgnosticFedAvgTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name':
              'EG',
          'domain_algorithm':
              agnostic_fed_avg.DomainAlgorithm.EG,
          'expected_domain_weights':
              [0.08078121, 0.26637138, 0.10979304, 0.54305437]
      }, {
          'testcase_name': 'NONE',
          'domain_algorithm': agnostic_fed_avg.DomainAlgorithm.NONE,
          'expected_domain_weights': [0.2, 0.4, 0.1, 0.3]
      })
  def test_update_domain_weights(self, domain_algorithm,
                                 expected_domain_weights):
    domain_weights = agnostic_fed_avg._update_domain_weights(
        domain_weights=jnp.array([0.2, 0.4, 0.1, 0.3]),
        domain_loss=jnp.array([1., 2., 3., 4.]),
        domain_learning_rate=0.5,
        domain_algorithm=domain_algorithm)

    self.assertAllClose(domain_weights, expected_domain_weights)

  def test_run(self):
    """Tests agnostic federated averaging on minmax regression.

    fedjax.datasets.toy_regression has the formal description.
    To summarize:
      1. Random points are generated along a line and assigned to 2 domains.
      2. The points in the two domains are offset set to have the same average
        value magnitude but opposite signs.
        (e.g. mean(domain_1) == -1.0 and mean(domain_2) == 1.0)
      3. Training task is to find the point value that minimizes the maximum
        distance from the domain centers. Given (2), this value should be 0.
    """
    federated_data, _ = toy_regression_data.load_data(
        num_clients=2, num_domains=2, num_points=400, seed=0)
    model = toy_regression_model.create_regression_model()
    rng_seq = core.PRNGSequence(0)
    algorithm = agnostic_fed_avg.AgnosticFedAvg(
        federated_data=federated_data,
        model=model,
        client_optimizer=core.get_optimizer(
            core.OptimizerName.SGD, learning_rate=0.3),
        # NB: Momentum doesn't play well with this toy example.
        server_optimizer=core.get_optimizer(
            core.OptimizerName.SGD, learning_rate=1.0),
        hparams=agnostic_fed_avg.AgnosticFedAvgHParams(
            # Batch size should be large enough to fit all data in one batch.
            # This is because toy_regression loss isn't an average over examples
            # meaning the multi batch evaluation logic will result in incorrect
            # "total loss" values.
            train_data_hparams=core.ClientDataHParams(batch_size=1000),
            init_domain_weights=(0.4, 0.6),
            domain_id_fn=toy_regression_data.domain_id_fn,
            domain_learning_rate=0.3,
            domain_algorithm=agnostic_fed_avg.DomainAlgorithm.EG,
        ),
        rng_seq=rng_seq,
    )

    state = algorithm.init_state()
    for _ in range(100):
      state = algorithm.run_round(state, federated_data.client_ids)

    # 0 should be the center between the max and min point averages.
    self.assertAlmostEqual(state.params['linear']['w'], 0., places=6)
    # Domain weights should even out.
    self.assertAllClose(state.domain_weights, [0.5, 0.5])


if __name__ == '__main__':
  tf.test.main()
