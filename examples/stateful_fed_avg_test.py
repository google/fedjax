# Copyright 2023 Google LLC
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
"""Tests for fedjax.examples.stateful_fed_avg."""
from absl.testing import absltest

import fedjax
import stateful_fed_avg

import jax
import jax.numpy as jnp
import numpy.testing as npt


def grad_fn(params, batch, rng):
  del rng
  return jax.tree_util.tree_map(lambda l: l / jnp.sum(batch['x']), params)


class StatefulFedAvgTest(absltest.TestCase):

  def test_stateful_federated_averaging(self):
    client_optimizer = fedjax.optimizers.sgd(learning_rate=1.0)
    server_optimizer = fedjax.optimizers.sgd(learning_rate=1.0)
    client_batch_hparams = fedjax.ShuffleRepeatBatchHParams(
        batch_size=2, num_epochs=1, seed=0)
    algorithm = stateful_fed_avg.stateful_federated_averaging(
        grad_fn, client_optimizer, server_optimizer, client_batch_hparams
    )

    with self.subTest('init'):
      state = algorithm.init({'w': jnp.array([0., 2., 4.])})
      npt.assert_array_equal(state.params['w'], [0., 2., 4.])
      self.assertLen(state.opt_state, 2)
      self.assertDictEqual(state.client_states, {})

    with self.subTest('apply'):
      clients = [
          (b'cid0',
           fedjax.ClientDataset({'x': jnp.array([2., 4., 6.])}),
           jax.random.PRNGKey(0)),
          (b'cid1',
           fedjax.ClientDataset({'x': jnp.array([8., 10.])}),
           jax.random.PRNGKey(1)),
      ]
      state, client_diagnostics = algorithm.apply(state, clients)
      npt.assert_allclose(state.params['w'], [0., 1.5655555, 3.131111])
      npt.assert_allclose(client_diagnostics[b'cid0']['delta_l2_norm'],
                          1.4534444262)
      npt.assert_allclose(client_diagnostics[b'cid1']['delta_l2_norm'],
                          0.2484521282)
      with self.subTest('client_states'):
        npt.assert_array_equal(state.client_states[b'cid0'].num_steps, 2)
        npt.assert_array_equal(state.client_states[b'cid1'].num_steps, 1)
        state, _ = algorithm.apply(state, clients)
        npt.assert_array_equal(state.client_states[b'cid0'].num_steps, 4)
        npt.assert_array_equal(state.client_states[b'cid1'].num_steps, 2)

  def test_create_train_for_each_client(self):
    client_optimizer = fedjax.optimizers.sgd(learning_rate=1.0)
    train_for_each_client = stateful_fed_avg.create_train_for_each_client(
        grad_fn, client_optimizer
    )
    batched_clients = [
        (
            # Client id.
            b'cid0',
            # Client batched dataset.
            [
                {'x': jnp.array([2.0, 4.0, 6.0])},
                {'x': jnp.array([8.0, 10.0, 12.0])},
            ],
            # Client input.
            {
                'rng': jax.random.PRNGKey(0),
                'state': stateful_fed_avg.ClientState(num_steps=2),
            },
        ),
        (
            b'cid1',
            [
                {'x': jnp.array([1.0, 3.0, 5.0])},
                {'x': jnp.array([7.0, 9.0, 11.0])},
            ],
            {
                'rng': jax.random.PRNGKey(1),
                'state': stateful_fed_avg.ClientState(num_steps=7),
            },
        ),
    ]
    server_params = {'w': jnp.array(4.0)}
    client_outputs = dict(train_for_each_client(server_params, batched_clients))
    npt.assert_allclose(client_outputs[b'cid0']['delta_params']['w'], 0.4555554)
    npt.assert_allclose(client_outputs[b'cid1']['delta_params']['w'], 0.5761316)
    npt.assert_array_equal(client_outputs[b'cid0']['state'].num_steps, 4)
    npt.assert_array_equal(client_outputs[b'cid1']['state'].num_steps, 9)


if __name__ == '__main__':
  absltest.main()
