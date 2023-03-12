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
"""Tests for fedjax.algorithms.apfl."""

from absl.testing import absltest

from fedjax.algorithms import apfl
from fedjax.core import client_datasets
from fedjax.core import metrics
from fedjax.core import models
from fedjax.core import optimizers

import jax
import jax.numpy as jnp
import numpy.testing as npt


def grad_fn(params, batch, rng):
  del rng
  return jax.tree_util.tree_map(lambda l: l / jnp.sum(batch['x']), params)


def fake_model():
  def apply_for_eval(params, example):
    return jax.nn.one_hot(example['x'] * params['w'] % 3, 3)

  eval_metrics = {'accuracy': metrics.Accuracy()}

  return models.Model(
    init=None,
    apply_for_train=None,
    apply_for_eval=apply_for_eval,
    train_loss=None,
    eval_metrics=eval_metrics)


class APFLTest(absltest.TestCase):

  def test_adaptive_personalized_federated_learning(self):
    algorithm = apfl.adaptive_personalized_federated_learning(
      grad_fn=grad_fn,
      client_optimizer=optimizers.sgd(learning_rate=1.0),
      server_optimizer=optimizers.sgd(learning_rate=1.0),
      client_coefficient=0.5,
      client_batch_hparams=client_datasets.ShuffleRepeatBatchHParams(
        batch_size=2,
        num_epochs=1,
        seed=0))

    with self.subTest('init'):
      params = {'w': jnp.array([0., 2., 4.])}
      state = algorithm.init(params)
      npt.assert_array_equal(state.params['w'], params['w'])
      self.assertLen(state.opt_state, 2)
      self.assertEmpty(state.client_states)

    with self.subTest('apply'):
      clients = [
        (b'cid0',
          client_datasets.ClientDataset({'x': jnp.array([2., 4., 6.])}),
          jax.random.PRNGKey(0)),
        (b'cid1',
          client_datasets.ClientDataset({'x': jnp.array([8., 10.])}),
          jax.random.PRNGKey(1))
      ]
      state, client_diagnostics = algorithm.apply(state, clients)
      npt.assert_allclose(
        state.params['w'],
        [0., 1.5655555, 3.131111])
      npt.assert_allclose(
        state.client_states[b'cid0'].params['w'],
        [0. , 1.35, 2.7])
      npt.assert_allclose(
        state.client_states[b'cid0'].interpolation_coefficients['w'],
        [0.5])
      npt.assert_allclose(
        state.client_states[b'cid1'].params['w'],
        [0., 1.888889, 3.777778])
      npt.assert_allclose(
        state.client_states[b'cid1'].interpolation_coefficients['w'],
        [0.5])
      npt.assert_allclose(
        client_diagnostics[b'cid0']['delta_l2_norm'],
        1.4534444262)
      npt.assert_allclose(
        client_diagnostics[b'cid1']['delta_l2_norm'],
        0.2484521282)

  def test_create_train_for_each_client(self):
    train_for_each_client = apfl.create_train_for_each_client(
      grad_fn=grad_fn,
      client_optimizer=optimizers.sgd(learning_rate=1.0))
    server_params = {'w': jnp.array(4.0)}
    batched_clients = [
        (b'cid0',
         [{'x': jnp.array([2., 4., 6.])}],
         {'rng': jax.random.PRNGKey(0),
          'state': apfl.ClientState(
            params={'w': jnp.array(2.0)},
            interpolation_coefficients={'w': jnp.array([0.])})  
        }),
        (b'cid1',
         [{'x': jnp.array([2., 4., 6.])}],
         {'rng': jax.random.PRNGKey(1),
          'state': apfl.ClientState(
            params={'w': jnp.array(2.0)},
            interpolation_coefficients={'w': jnp.array([0.5])})
        }),
        (b'cid2',
         [{'x': jnp.array([2., 4., 6.])}],
         {'rng': jax.random.PRNGKey(2),
          'state': apfl.ClientState(
            params={'w': jnp.array(2.0)},
            interpolation_coefficients={'w': jnp.array([1.])})
        }),
        (b'cid3',
         [{'x': jnp.array([2., 4., 6.])}],
         {'rng': jax.random.PRNGKey(3),
          'state': apfl.ClientState(
            params={'w': jnp.array(8.0)},
            interpolation_coefficients={'w': jnp.array([0.])})  
        }),
        (b'cid4',
         [{'x': jnp.array([2., 4., 6.])}],
         {'rng': jax.random.PRNGKey(4),
          'state': apfl.ClientState(
            params={'w': jnp.array(8.0)},
            interpolation_coefficients={'w': jnp.array([0.5])})
        }),
        (b'cid5',
         [{'x': jnp.array([2., 4., 6.])}],
         {'rng': jax.random.PRNGKey(5),
          'state': apfl.ClientState(
            params={'w': jnp.array(8.0)},
            interpolation_coefficients={'w': jnp.array([1.])})
        }),
    ]
    client_outputs = dict(train_for_each_client(server_params, batched_clients))
    npt.assert_allclose(client_outputs[b'cid0']['delta_params']['w'], 0.33333325)
    npt.assert_allclose(client_outputs[b'cid0']['state'].params['w'], 1.6666666)
    npt.assert_allclose(client_outputs[b'cid0']['state'].interpolation_coefficients['w'], 0.6666667)
    npt.assert_allclose(client_outputs[b'cid1']['delta_params']['w'], 0.33333325)
    npt.assert_allclose(client_outputs[b'cid1']['state'].params['w'], 1.75)
    npt.assert_allclose(client_outputs[b'cid1']['state'].interpolation_coefficients['w'], 1.)
    npt.assert_allclose(client_outputs[b'cid2']['delta_params']['w'], 0.33333325)
    npt.assert_allclose(client_outputs[b'cid2']['state'].params['w'], 1.8333334)
    npt.assert_allclose(client_outputs[b'cid2']['state'].interpolation_coefficients['w'], 1.)
    npt.assert_allclose(client_outputs[b'cid3']['delta_params']['w'], 0.33333325)
    npt.assert_allclose(client_outputs[b'cid3']['state'].params['w'], 7.6666665)
    npt.assert_allclose(client_outputs[b'cid3']['state'].interpolation_coefficients['w'], 0.)
    npt.assert_allclose(client_outputs[b'cid4']['delta_params']['w'], 0.33333325)
    npt.assert_allclose(client_outputs[b'cid4']['state'].params['w'], 7.5)
    npt.assert_allclose(client_outputs[b'cid4']['state'].interpolation_coefficients['w'], 0.)
    npt.assert_allclose(client_outputs[b'cid5']['delta_params']['w'], 0.33333325)
    npt.assert_allclose(client_outputs[b'cid5']['state'].params['w'], 7.3333335)
    npt.assert_allclose(client_outputs[b'cid5']['state'].interpolation_coefficients['w'], 0.)

  def test_eval_adaptive_personalized_federated_learning(self):
    eval_fn = apfl.eval_adaptive_personalized_federated_learning(
      model=fake_model(),
      client_batch_hparams=client_datasets.PaddedBatchHParams(batch_size=3))

    state = apfl.ServerState(
      params={'w': jnp.array([0., 2., 4.])},
      opt_state=None,
      client_states={
        b'cid0': apfl.ClientState(
          params={'w': jnp.array([0., 1., 2.])},
          interpolation_coefficients={'w': jnp.array([0.])}),
        b'cid1': apfl.ClientState(
          params={'w': jnp.array([0., 1., 2.])},
          interpolation_coefficients={'w': jnp.array([0.5])}),
        b'cid2': apfl.ClientState(
          params={'w': jnp.array([0., 1., 2.])},
          interpolation_coefficients={'w': jnp.array([1.])})
      })

    clients = [
      (b'cid0',
        client_datasets.ClientDataset(
          {'x': jnp.array([0., 2., 4.]), 'y': jnp.array([0, 2, 2])})),
      (b'cid1',
        client_datasets.ClientDataset(
          {'x': jnp.array([0., 2., 4.]), 'y': jnp.array([0, 2, 2])})),
      (b'cid2',
        client_datasets.ClientDataset(
          {'x': jnp.array([0., 2., 4.]), 'y': jnp.array([0, 2, 2])})),
      (b'cid3',
        client_datasets.ClientDataset(
          {'x': jnp.array([0., 2., 4.]), 'y': jnp.array([0, 2, 2])})),
    ]
    client_outputs = dict(eval_fn(state, clients))
    npt.assert_allclose(client_outputs[b'cid0']['accuracy'].result(), 0.33333334)
    npt.assert_allclose(client_outputs[b'cid1']['accuracy'].result(), 0.33333334)
    npt.assert_allclose(client_outputs[b'cid2']['accuracy'].result(), 1.)
    npt.assert_allclose(client_outputs[b'cid3']['accuracy'].result(), 0.33333334)

  def test_create_eval_for_each_client(self):
    eval_for_each_client = apfl.create_eval_for_each_client(model=fake_model())
    server_params = {'w': jnp.array([0., 2., 4.])}
    batched_clients = [
        (b'cid0',
         [{'x': jnp.array([2., 4., 6.]), 'y': jnp.array([0., 1., 0.])},
          {'x': jnp.array([8., 10., 12.]), 'y': jnp.array([0., 1., 0.])}],
         apfl.ClientState(
          params={'w': jnp.array([0., 1., 2.])},
          interpolation_coefficients={'w': jnp.array([0.5])})
        ),
        (b'cid1',
         [{'x': jnp.array([1., 3., 5.]), 'y': jnp.array([1., 1., 1.])},
          {'x': jnp.array([7., 9., 11.]), 'y': jnp.array([1., 1., 1.])}],
         apfl.ClientState(
          params={'w': jnp.array([0., 4., 16.])},
          interpolation_coefficients={'w': jnp.array([0.5])})
        ),
    ]
    client_outputs = dict(eval_for_each_client(server_params, batched_clients))
    npt.assert_allclose(client_outputs[b'cid0']['accuracy'].result(), 0.6666667)
    npt.assert_allclose(client_outputs[b'cid1']['accuracy'].result(), 0)


if __name__ == '__main__':
  absltest.main()
