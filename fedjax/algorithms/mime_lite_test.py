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
"""Tests for fedjax.algorithms.mime_lite."""

from absl.testing import absltest

from fedjax.algorithms import mime_lite
from fedjax.core import client_datasets
from fedjax.core import models
from fedjax.core import optimizers

import jax
import jax.numpy as jnp
import numpy.testing as npt


def per_example_loss(params, batch, rng):
  del rng
  return batch['x'] * params['w']


grad_fn = models.grad(per_example_loss)


class MimeLiteTest(absltest.TestCase):

  def test_mime_lite(self):
    base_optimizer = optimizers.sgd(learning_rate=1.0)
    train_batch_hparams = client_datasets.ShuffleRepeatBatchHParams(
        batch_size=2, num_epochs=1, seed=0)
    grad_batch_hparams = client_datasets.PaddedBatchHParams(batch_size=2)
    server_learning_rate = 0.2
    algorithm = mime_lite.mime_lite(per_example_loss, base_optimizer,
                                    train_batch_hparams, grad_batch_hparams,
                                    server_learning_rate)

    with self.subTest('init'):
      state = algorithm.init({'w': jnp.array(4.)})
      npt.assert_equal(state.params, {'w': jnp.array(4.)})
      self.assertLen(state.opt_state, 2)

    with self.subTest('apply'):
      clients = [
          (b'cid0',
           client_datasets.ClientDataset({'x': jnp.array([0.2, 0.4, 0.6])}),
           jax.random.PRNGKey(0)),
          (b'cid1', client_datasets.ClientDataset({'x': jnp.array([0.8, 0.1])}),
           jax.random.PRNGKey(1)),
      ]
      state, client_diagnostics = algorithm.apply(state, clients)
      npt.assert_allclose(state.params['w'], 3.8799999)
      npt.assert_allclose(client_diagnostics[b'cid0']['delta_l2_norm'],
                          0.70000005)
      npt.assert_allclose(client_diagnostics[b'cid1']['delta_l2_norm'],
                          0.45000005)

  def test_create_train_for_each_client(self):
    base_optimizer = optimizers.sgd(learning_rate=1.0)
    train_for_each_client = mime_lite.create_train_for_each_client(
        grad_fn, base_optimizer)
    batch_clients = [
        (b'cid0', [{
            'x': jnp.array([0.6, 0.4])
        }, {
            'x': jnp.array([0.2, 0.2])
        }], jax.random.PRNGKey(0)),
        (b'cid1', [{
            'x': jnp.array([0.1, 0.8])
        }], jax.random.PRNGKey(1)),
    ]
    server_params = {'w': jnp.array(4.)}
    server_opt_state = base_optimizer.init(server_params)
    shared_input = {
        'params': server_params,
        'opt_state': server_opt_state,
    }
    client_outputs = dict(train_for_each_client(shared_input, batch_clients))
    npt.assert_allclose(client_outputs[b'cid0']['w'], 0.7)
    npt.assert_allclose(client_outputs[b'cid1']['w'], 0.45000002)

  def test_client_delta_clip_norm(self):
    base_optimizer = optimizers.sgd(learning_rate=1.0)
    train_batch_hparams = client_datasets.ShuffleRepeatBatchHParams(
        batch_size=2, num_epochs=1, seed=0)
    grad_batch_hparams = client_datasets.PaddedBatchHParams(batch_size=2)
    server_learning_rate = 0.2
    algorithm = mime_lite.mime_lite(
        per_example_loss,
        base_optimizer,
        train_batch_hparams,
        grad_batch_hparams,
        server_learning_rate,
        client_delta_clip_norm=0.5)

    clients = [
        (b'cid0',
         client_datasets.ClientDataset({'x': jnp.array([0.2, 0.4, 0.6])}),
         jax.random.PRNGKey(0)),
        (b'cid1', client_datasets.ClientDataset({'x': jnp.array([0.8, 0.1])}),
         jax.random.PRNGKey(1)),
    ]
    state = algorithm.init({'w': jnp.array(4.)})
    state, client_diagnostics = algorithm.apply(state, clients)
    npt.assert_allclose(state.params['w'], 3.904)
    npt.assert_allclose(client_diagnostics[b'cid0']['clipped_delta_l2_norm'],
                        0.5)
    npt.assert_allclose(client_diagnostics[b'cid1']['clipped_delta_l2_norm'],
                        0.45000005)


if __name__ == '__main__':
  absltest.main()
