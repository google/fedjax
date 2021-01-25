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
"""Tests for simple_fed_avg."""

import fedjax
import simple_fed_avg
import tensorflow as tf


class SimpleFedAvgTest(tf.test.TestCase):

  def test_run(self):
    federated_data, model = fedjax.test_util.create_toy_example(
        num_clients=10, num_clusters=4, num_classes=10, num_examples=5, seed=0)
    algorithm = simple_fed_avg.SimpleFedAvg(
        federated_data=federated_data,
        model=model,
        client_optimizer=fedjax.get_optimizer(
            fedjax.OptimizerName.SGD, learning_rate=0.1),
        server_optimizer=fedjax.get_optimizer(
            fedjax.OptimizerName.MOMENTUM, learning_rate=2.0, momentum=0.9),
        hparams=simple_fed_avg.SimpleFedAvgHParams(
            train_data_hparams=fedjax.ClientDataHParams(batch_size=100)),
        rng_seq=fedjax.PRNGSequence(0))

    state = algorithm.init_state()
    combined_dataset = fedjax.create_tf_dataset_for_clients(
        federated_data).batch(50)
    init_metrics = fedjax.evaluate_single_client(combined_dataset, model,
                                                 state.params)
    for _ in range(10):
      state = algorithm.run_round(state, federated_data.client_ids)

    metrics = fedjax.evaluate_single_client(combined_dataset, model,
                                            state.params)
    self.assertLess(metrics['loss'], init_metrics['loss'])
    self.assertGreater(metrics['accuracy'], init_metrics['accuracy'])


if __name__ == '__main__':
  tf.test.main()
