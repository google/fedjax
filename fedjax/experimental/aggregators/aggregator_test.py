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
"""Tests for fedjax.aggregators.aggregator."""

from fedjax import core
from fedjax.experimental.aggregators import aggregator
import tensorflow as tf


class AggregatorTest(tf.test.TestCase):

  def test_aggregator(self):
    num_classes = 10
    num_clients = 10
    num_examples = 5
    data, model = core.test_util.create_toy_example(
        num_clients=10,
        num_clusters=4,
        num_classes=num_classes,
        num_examples=5,
        seed=0)
    rng_seq = core.PRNGSequence(0)
    client_optimizer = core.get_optimizer(
        core.OptimizerName.SGD, learning_rate=0.1)
    client_trainer = core.DefaultClientTrainer(model, client_optimizer)
    init_params = model.init_params(0)
    client_outputs = core.train_multiple_clients(
        federated_data=data,
        client_ids=data.client_ids,
        client_trainer=client_trainer,
        init_client_trainer_state=client_trainer.init_state(init_params),
        rng_seq=rng_seq,
        client_data_hparams=core.ClientDataHParams(batch_size=10))

    def get_delta_params_and_weight(client_output):
      delta_params = core.tree_multimap(lambda a, b: a - b, init_params,
                                        client_output.params)
      return delta_params, client_output.num_examples

    delta_params_and_weight = map(get_delta_params_and_weight, client_outputs)

    default_aggregator = aggregator.MeanAggregator()
    init_aggregator_state = default_aggregator.init_state()
    rng_seq2 = core.PRNGSequence(1)
    _, new_state = default_aggregator.aggregate(init_aggregator_state,
                                                delta_params_and_weight,
                                                rng_seq2)
    self.assertEqual(new_state.total_weight, num_clients * num_examples)


if __name__ == '__main__':
  tf.test.main()
