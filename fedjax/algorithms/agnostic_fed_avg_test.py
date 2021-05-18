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
"""Tests for fedjax.algorithms.agnostic_fed_avg."""

from absl.testing import absltest

from fedjax.algorithms import agnostic_fed_avg
from fedjax.core import client_datasets
from fedjax.core import optimizers

import jax
import jax.numpy as jnp
import numpy.testing as npt


def per_example_loss(params, batch, rng):
  del rng
  return batch['x'] * params['w']


class AgnosticFedAvgTest(absltest.TestCase):

  def test_agnostic_federated_averaging(self):
    algorithm = agnostic_fed_avg.agnostic_federated_averaging(
        per_example_loss=per_example_loss,
        client_optimizer=optimizers.sgd(learning_rate=1.0),
        server_optimizer=optimizers.sgd(learning_rate=0.1),
        client_batch_hparams=client_datasets.ShuffleRepeatBatchHParams(
            batch_size=3, num_epochs=1, seed=0),
        domain_batch_hparams=client_datasets.PaddedBatchHParams(batch_size=3),
        init_domain_weights=[0.1, 0.2, 0.3, 0.4],
        domain_learning_rate=0.01,
        domain_algorithm='eg',
        domain_window_size=2,
        init_domain_window=[1., 2., 3., 4.])

    with self.subTest('init'):
      state = algorithm.init({'w': jnp.array(4.)})
      npt.assert_equal(state.params, {'w': jnp.array(4.)})
      self.assertLen(state.opt_state, 2)
      npt.assert_allclose(state.domain_weights, [0.1, 0.2, 0.3, 0.4])
      npt.assert_allclose(state.domain_window,
                          [[1., 2., 3., 4.], [1., 2., 3., 4.]])

    with self.subTest('apply'):
      clients = [
          (b'cid0',
           client_datasets.ClientDataset({
               'x': jnp.array([1., 2., 4., 3., 6., 1.]),
               'domain_id': jnp.array([1, 0, 0, 0, 2, 2])
           }), jax.random.PRNGKey(0)),
          (b'cid1',
           client_datasets.ClientDataset({
               'x': jnp.array([8., 10., 5.]),
               'domain_id': jnp.array([1, 3, 1])
           }), jax.random.PRNGKey(1)),
      ]
      next_state, client_diagnostics = algorithm.apply(state, clients)
      npt.assert_allclose(next_state.params['w'], 3.5555556)
      npt.assert_allclose(next_state.domain_weights,
                          [0.08702461, 0.18604803, 0.2663479, 0.46057943])
      npt.assert_allclose(next_state.domain_window,
                          [[1., 2., 3., 4.], [3., 3., 2., 1.]])
      npt.assert_allclose(client_diagnostics[b'cid0']['delta_l2_norm'],
                          2.8333335)
      npt.assert_allclose(client_diagnostics[b'cid1']['delta_l2_norm'],
                          7.666667)

    with self.subTest('invalid init_domain_weights'):
      with self.assertRaisesRegex(
          ValueError, 'init_domain_weights must sum to approximately 1.'):
        agnostic_fed_avg.agnostic_federated_averaging(
            per_example_loss=per_example_loss,
            client_optimizer=optimizers.sgd(learning_rate=1.0),
            server_optimizer=optimizers.sgd(learning_rate=1.0),
            client_batch_hparams=client_datasets.ShuffleRepeatBatchHParams(
                batch_size=3),
            domain_batch_hparams=client_datasets.PaddedBatchHParams(
                batch_size=3),
            init_domain_weights=[50., 0., 0., 0.],
            domain_learning_rate=0.5)

    with self.subTest('unequal lengths'):
      with self.assertRaisesRegex(
          ValueError,
          'init_domain_weights and init_domain_window must be equal lengths.'):
        agnostic_fed_avg.agnostic_federated_averaging(
            per_example_loss=per_example_loss,
            client_optimizer=optimizers.sgd(learning_rate=1.0),
            server_optimizer=optimizers.sgd(learning_rate=1.0),
            client_batch_hparams=client_datasets.ShuffleRepeatBatchHParams(
                batch_size=3),
            domain_batch_hparams=client_datasets.PaddedBatchHParams(
                batch_size=3),
            init_domain_weights=[0.1, 0.2, 0.3, 0.4],
            domain_learning_rate=0.5,
            init_domain_window=[1, 2])

  def test_update_domain_weights(self):
    domain_weights = jnp.array([0.1, 0.2, 0.3, 0.4])
    domain_loss = jnp.array([10., 0., 5., 0.])
    domain_learning_rate = 0.1

    with self.subTest('eg'):
      npt.assert_allclose(
          agnostic_fed_avg.update_domain_weights(
              domain_weights,
              domain_loss,
              domain_learning_rate,
              domain_algorithm='eg'),
          [0.198931, 0.14636526, 0.36197326, 0.2927305])

    with self.subTest('none'):
      npt.assert_allclose(
          agnostic_fed_avg.update_domain_weights(
              domain_weights,
              domain_loss,
              domain_learning_rate,
              domain_algorithm='none'), [0.1, 0.2, 0.3, 0.4])

    with self.subTest('INVALID'):
      with self.assertRaisesRegex(ValueError, 'Unsupported domain algorithm'):
        agnostic_fed_avg.update_domain_weights(
            domain_weights,
            domain_loss,
            domain_learning_rate,
            domain_algorithm='INVALID')

  def test_create_domain_metrics_for_each_client(self):
    num_domains = 4
    shared_input = {
        'params': {
            'w': jnp.array(4.)
        },
        'alpha': jnp.array([0.1, 0.2, 0.3, 0.4])
    }
    padded_batch_clients = [
        (b'cid0', [{
            'x': jnp.array([1., 2., 4.]),
            'domain_id': jnp.array([1, 0, 0]),
            '__mask__': jnp.array([True, True, True])
        }, {
            'x': jnp.array([3., 6., 1.]),
            'domain_id': jnp.array([0, 2, 2]),
            '__mask__': jnp.array([True, True, False])
        }], jax.random.PRNGKey(0)),
        (b'cid1', [{
            'x': jnp.array([8., 10., 5.]),
            'domain_id': jnp.array([1, 3, 1]),
            '__mask__': jnp.array([True, True, False])
        }], jax.random.PRNGKey(1)),
    ]
    func = agnostic_fed_avg.create_domain_metrics_for_each_client(
        per_example_loss, num_domains)
    client_domain_metrics = dict(func(shared_input, padded_batch_clients))
    npt.assert_allclose(client_domain_metrics[b'cid0']['beta'], 0.8)
    npt.assert_allclose(client_domain_metrics[b'cid0']['domain_loss'],
                        [36., 4., 24., 0.])
    npt.assert_allclose(client_domain_metrics[b'cid0']['domain_num'],
                        [3., 1., 1., 0.])
    npt.assert_allclose(client_domain_metrics[b'cid1']['beta'], 0.6)
    npt.assert_allclose(client_domain_metrics[b'cid1']['domain_loss'],
                        [0., 32., 0., 40.])
    npt.assert_allclose(client_domain_metrics[b'cid1']['domain_num'],
                        [0., 1., 0., 1.])

  def test_create_scaled_loss(self):
    num_domains = 4
    scaled_loss = agnostic_fed_avg.create_scaled_loss(per_example_loss,
                                                      num_domains)
    params = {'w': jnp.array(4.)}
    batch = {
        'x': jnp.array([1., 2., 4.]),
        'domain_id': jnp.array([1, 0, 0]),
    }
    rng = jax.random.PRNGKey(0)
    alpha = jnp.array([0.1, 0.2, 0.3, 0.4])
    beta = 0.5
    npt.assert_allclose(scaled_loss(params, batch, rng, alpha, beta), 6.4)

  def test_create_train_for_each_client(self):
    num_domains = 4
    shared_input = {
        'params': {
            'w': jnp.array(4.)
        },
        'alpha': jnp.array([0.1, 0.2, 0.3, 0.4])
    }
    batch_clients = [
        (b'cid0', [{
            'x': jnp.array([1., 2., 4.]),
            'domain_id': jnp.array([1, 0, 0]),
        }, {
            'x': jnp.array([3., 6., 1.]),
            'domain_id': jnp.array([0, 2, 2]),
        }], {
            'rng': jax.random.PRNGKey(0),
            'beta': jnp.array(0.5)
        }),
        (b'cid1', [{
            'x': jnp.array([8., 10., 5.]),
            'domain_id': jnp.array([1, 3, 1]),
        }], {
            'rng': jax.random.PRNGKey(1),
            'beta': jnp.array(0.2)
        }),
    ]
    client_optimizer = optimizers.sgd(learning_rate=1.0)
    func = agnostic_fed_avg.create_train_for_each_client(
        per_example_loss, client_optimizer, num_domains)
    client_delta_params = dict(func(shared_input, batch_clients))
    npt.assert_allclose(client_delta_params[b'cid0']['w'], 6.4)
    npt.assert_allclose(client_delta_params[b'cid1']['w'], 33.)


if __name__ == '__main__':
  absltest.main()
