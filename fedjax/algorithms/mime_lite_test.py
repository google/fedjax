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
"""Tests for core.algorithms.mime_lite."""

from fedjax import core
from fedjax.algorithms import mime_lite
import tensorflow as tf


class MimeLiteTest(tf.test.TestCase):

  def test_run(self):
    num_classes = 10
    data, model = core.test_util.create_toy_example(
        num_clients=10,
        num_clusters=4,
        num_classes=num_classes,
        num_examples=5,
        seed=0)
    rng_seq = core.PRNGSequence(0)
    algorithm = mime_lite.MimeLite(
        federated_data=data,
        model=model,
        base_optimizer=core.get_optimizer(
            core.OptimizerName.MOMENTUM, learning_rate=0.2, momentum=0.9),
        hparams=mime_lite.MimeLiteHParams(
            train_data_hparams=core.ClientDataHParams(batch_size=100),
            combined_data_hparams=core.ClientDataHParams(batch_size=100),
            server_learning_rate=1.0),
        rng_seq=rng_seq,
    )
    dataset = core.create_tf_dataset_for_clients(data).batch(50)

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
