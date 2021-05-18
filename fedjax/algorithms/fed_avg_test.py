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
"""Tests for fedjax.algorithms.fed_avg."""
from absl.testing import absltest

from fedjax.algorithms import fed_avg
from fedjax.core import client_datasets
from fedjax.core import optimizers

import jax
import jax.numpy as jnp
import numpy.testing as npt


def grad_fn(params, batch, rng):
  del rng
  return jax.tree_util.tree_map(lambda l: l / jnp.sum(batch['x']), params)


class FedAvgTest(absltest.TestCase):

  def test_federated_averaging(self):
    client_optimizer = optimizers.sgd(learning_rate=1.0)
    server_optimizer = optimizers.sgd(learning_rate=1.0)
    client_batch_hparams = client_datasets.ShuffleRepeatBatchHParams(
        batch_size=2, num_epochs=1, seed=0)
    algorithm = fed_avg.federated_averaging(grad_fn, client_optimizer,
                                            server_optimizer,
                                            client_batch_hparams)

    with self.subTest('init'):
      state = algorithm.init({'w': jnp.array([0., 2., 4.])})
      npt.assert_array_equal(state.params['w'], [0., 2., 4.])
      self.assertLen(state.opt_state, 2)

    with self.subTest('apply'):
      clients = [
          (b'cid0',
           client_datasets.ClientDataset({'x': jnp.array([2., 4., 6.])}),
           jax.random.PRNGKey(0)),
          (b'cid1',
           client_datasets.ClientDataset({'x': jnp.array([8., 10.])}),
           jax.random.PRNGKey(1)),
      ]
      state, client_diagnostics = algorithm.apply(state, clients)
      npt.assert_allclose(state.params['w'], [0., 1.5655555, 3.131111])
      npt.assert_allclose(client_diagnostics[b'cid0']['delta_l2_norm'],
                          1.4534444262)
      npt.assert_allclose(client_diagnostics[b'cid1']['delta_l2_norm'],
                          0.2484521282)

  def test_create_train_for_each_client(self):
    client_optimizer = optimizers.sgd(learning_rate=1.0)
    train_for_each_client = fed_avg.create_train_for_each_client(
        grad_fn, client_optimizer)
    batched_clients = [
        (b'cid0',
         [{'x': jnp.array([2., 4., 6.])}, {'x': jnp.array([8., 10., 12.])}],
         jax.random.PRNGKey(0)),
        (b'cid1',
         [{'x': jnp.array([1., 3., 5.])}, {'x': jnp.array([7., 9., 11.])}],
         jax.random.PRNGKey(1)),
    ]
    server_params = {'w': jnp.array(4.0)}
    client_outputs = dict(train_for_each_client(server_params, batched_clients))
    npt.assert_allclose(client_outputs[b'cid0']['w'], 0.45555544)
    npt.assert_allclose(client_outputs[b'cid1']['w'], 0.5761316)


if __name__ == '__main__':
  absltest.main()
