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
"""Tests for fedjax.algorithms.mime."""

from absl.testing import absltest

from fedjax.algorithms import mime
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


class MimeTest(absltest.TestCase):

  def test_mime(self):
    base_optimizer = optimizers.sgd(learning_rate=1.0)
    train_batch_hparams = client_datasets.ShuffleRepeatBatchHParams(
        batch_size=2, num_epochs=1, seed=0)
    grad_batch_hparams = client_datasets.PaddedBatchHParams(batch_size=2)
    server_learning_rate = 0.2
    algorithm = mime.mime(per_example_loss, base_optimizer, train_batch_hparams,
                          grad_batch_hparams, server_learning_rate)

    with self.subTest('init'):
      state = algorithm.init({'w': jnp.array(4.)})
      npt.assert_equal(state.params, {'w': jnp.array(4.)})
      self.assertLen(state.opt_state, 2)

    with self.subTest('apply'):
      clients = [
          (b'cid0',
           client_datasets.ClientDataset({'x': jnp.array([2., 4., 6.])}),
           jax.random.PRNGKey(0)),
          (b'cid1', client_datasets.ClientDataset({'x': jnp.array([8., 10.])}),
           jax.random.PRNGKey(1)),
      ]
      state, client_diagnostics = algorithm.apply(state, clients)
      npt.assert_allclose(state.params['w'], 2.08)
      npt.assert_allclose(client_diagnostics[b'cid0']['delta_l2_norm'], 12.)
      npt.assert_allclose(client_diagnostics[b'cid1']['delta_l2_norm'], 6.)

  def test_create_grads_for_each_client(self):
    grads_for_each_client = mime.create_grads_for_each_client(grad_fn)
    server_params = {'w': jnp.array(4.)}
    padded_batch_clients = [
        (b'cid0', [{
            'x': jnp.array([2., 4.]),
            '__mask__': jnp.array([True, True])
        }, {
            'x': jnp.array([6., 0.]),
            '__mask__': jnp.array([True, False])
        }], jax.random.PRNGKey(0)),
        (b'cid1', [{
            'x': jnp.array([8., 10.]),
            '__mask__': jnp.array([True, True])
        }], jax.random.PRNGKey(1)),
    ]
    npt.assert_equal(
        list(grads_for_each_client(server_params, padded_batch_clients)),
        [(b'cid0', ({'w': jnp.array(12.)}, jnp.array(3.))),
         (b'cid1', ({'w': jnp.array(18.)}, jnp.array(2.)))])

  def test_create_train_for_each_client(self):
    base_optimizer = optimizers.sgd(learning_rate=1.0)
    train_for_each_client = mime.create_train_for_each_client(
        grad_fn, base_optimizer)
    batch_clients = [
        (b'cid0', [{
            'x': jnp.array([6., 4.])
        }, {
            'x': jnp.array([2., 2.])
        }], jax.random.PRNGKey(0)),
        (b'cid1', [{
            'x': jnp.array([10., 8])
        }], jax.random.PRNGKey(1)),
    ]
    server_params = {'w': jnp.array(4.)}
    server_opt_state = base_optimizer.init(server_params)
    server_grads = {'w': jnp.array(6.)}
    shared_input = {
        'params': server_params,
        'opt_state': server_opt_state,
        'control_variate': server_grads
    }
    client_outputs = dict(train_for_each_client(shared_input, batch_clients))
    npt.assert_allclose(client_outputs[b'cid0']['w'], 12.)
    npt.assert_allclose(client_outputs[b'cid1']['w'], 6.)


if __name__ == '__main__':
  absltest.main()
