# Copyright 2022 Google LLC
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
"""Tests for fedjax.algorithms.fed_prox."""
from absl.testing import absltest

from fedjax.algorithms import fed_prox
from fedjax.core import client_datasets
from fedjax.core import optimizers
from fedjax.core import tree_util

import jax
import jax.numpy as jnp
import numpy.testing as npt


def per_example_loss(params, batch, rng):
  del rng
  print(batch['x'])
  print(params['w'])
  return batch['x'] * params['w']


class FedProxTest(absltest.TestCase):

  def test_fed_prox(self):
    client_optimizer = optimizers.sgd(learning_rate=1.0)
    server_optimizer = optimizers.sgd(learning_rate=1.0)
    client_batch_hparams = client_datasets.ShuffleRepeatBatchHParams(
        batch_size=2, num_epochs=1, seed=0)
    algorithm = fed_prox.fed_prox(
        per_example_loss,
        client_optimizer,
        server_optimizer,
        client_batch_hparams,
        proximal_weight=0.01)

    with self.subTest('init'):
      state = algorithm.init({'w': jnp.array(4.)})
      npt.assert_array_equal(state.params['w'], 4.)
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
      npt.assert_allclose(state.params['w'], -3.77)
      npt.assert_allclose(client_diagnostics[b'cid0']['delta_l2_norm'], 6.95)
      npt.assert_allclose(client_diagnostics[b'cid1']['delta_l2_norm'], 9.)

  def test_create_train_for_each_client(self):
    proximal_weight = 0.01

    def fed_prox_loss(params, server_params, batch, rng):
      example_loss = per_example_loss(params, batch, rng)
      proximal_loss = 0.5 * proximal_weight * tree_util.tree_l2_squared(
          jax.tree_util.tree_map(lambda a, b: a - b, server_params,
                                      params))
      return jnp.mean(example_loss + proximal_loss)

    grad_fn = jax.grad(fed_prox_loss)
    client_optimizer = optimizers.sgd(learning_rate=1.0)
    train_for_each_client = fed_prox.create_train_for_each_client(
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
    npt.assert_allclose(client_outputs[b'cid0']['w'], 13.96)
    npt.assert_allclose(client_outputs[b'cid1']['w'], 11.97)


if __name__ == '__main__':
  absltest.main()
