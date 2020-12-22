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
"""Tests for fedjax.algorithm.fed_avg."""

from fedjax import core
from fedjax.algorithms import fed_avg
import tensorflow as tf


class FedAvgTest(tf.test.TestCase):

  def test_run(self):
    data, model = core.test_util.create_toy_example(
        num_clients=10, num_clusters=4, num_classes=10, num_examples=5, seed=0)
    dataset = core.create_tf_dataset_for_clients(data).batch(50)
    algorithm = fed_avg.FedAvg(
        federated_data=data,
        model=model,
        client_optimizer=core.get_optimizer(
            core.OptimizerName.SGD, learning_rate=0.1),
        server_optimizer=core.get_optimizer(
            core.OptimizerName.MOMENTUM, learning_rate=2.0, momentum=0.9),
        hparams=fed_avg.FedAvgHParams(
            train_data_hparams=core.ClientDataHParams(
                batch_size=100, num_epochs=1)),
        rng_seq=core.PRNGSequence(0))

    state = algorithm.init_state()
    init_metrics = core.evaluate_single_client(
        dataset=dataset, model=model, params=state.params)
    for _ in range(10):
      state = algorithm.run_round(state, data.client_ids)

    metrics = core.evaluate_single_client(
        dataset=dataset, model=model, params=state.params)
    self.assertLess(metrics['loss'], init_metrics['loss'])
    self.assertGreater(metrics['accuracy'], init_metrics['accuracy'])


if __name__ == '__main__':
  tf.test.main()
