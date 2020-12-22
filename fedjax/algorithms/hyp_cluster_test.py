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
"""Tests for fedjax.algorithms.hyp_cluster."""

from fedjax import core
from fedjax.algorithms import hyp_cluster
import tensorflow as tf


class HypClusterTest(tf.test.TestCase):

  def test_run(self):
    num_classes = 10
    num_clusters = 3
    federated_data, model = core.test_util.create_toy_example(
        num_clients=10,
        num_clusters=num_clusters,
        num_classes=num_classes,
        num_examples=5,
        seed=0)
    rng_seq = core.PRNGSequence(0)
    algorithm = hyp_cluster.HypCluster(
        federated_data=federated_data,
        model=model,
        client_optimizer=core.get_optimizer(
            core.OptimizerName.SGD, learning_rate=0.1),
        server_optimizer=core.get_optimizer(
            core.OptimizerName.SGD, learning_rate=1.0),
        hparams=hyp_cluster.HypClusterHParams(
            train_data_hparams=core.ClientDataHParams(batch_size=5),
            num_clusters=num_clusters),
        rng_seq=rng_seq,
    )

    state = algorithm.init_state()
    for _ in range(10):
      state = algorithm.run_round(state, federated_data.client_ids)

    with self.subTest('num_clusters'):
      self.assertLen(state.cluster_params, num_clusters)

    with self.subTest('maximization'):
      data_hparams = core.ClientDataHParams(batch_size=5)
      cluster_client_ids = hyp_cluster.maximization(federated_data,
                                                    federated_data.client_ids,
                                                    model, state.cluster_params,
                                                    data_hparams)
      for cluster_id, client_ids in enumerate(cluster_client_ids):
        for client_id in client_ids:
          dataset = federated_data.create_tf_dataset_for_client(client_id)
          dataset = core.preprocess_tf_dataset(dataset, data_hparams)
          cluster_loss = []
          for params in state.cluster_params:
            cluster_loss.append(
                core.evaluate_single_client(dataset, model, params)['loss'])
          # Cluster should be the best params for a client because clients are
          # clustered based on empirical loss.
          self.assertEqual(cluster_id, cluster_loss.index(min(cluster_loss)))


if __name__ == '__main__':
  tf.test.main()
