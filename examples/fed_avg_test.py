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
"""Tests for fedjax.examples.fed_avg."""
from absl.testing import absltest

import fedjax
import fed_avg

import jax
import jax.numpy as jnp
import numpy.testing as npt


def grad_fn(params, batch, rng):
  del rng
  return jax.tree_util.tree_map(lambda l: l / jnp.sum(batch['x']), params)


class FedAvgTest(absltest.TestCase):

  def test_federated_averaging(self):
    client_optimizer = fedjax.optimizers.sgd(learning_rate=1.0)
    server_optimizer = fedjax.optimizers.sgd(learning_rate=1.0)
    client_batch_hparams = fedjax.ShuffleRepeatBatchHParams(
        batch_size=2, num_epochs=2, seed=0)
    algorithm = fed_avg.federated_averaging(grad_fn, client_optimizer,
                                            server_optimizer,
                                            client_batch_hparams)
    with self.subTest('init'):
      state = algorithm.init({'w': jnp.array([0., 2., 4.])})
      npt.assert_array_equal(state.params['w'], [0., 2., 4.])
      self.assertLen(state.opt_state, 2)
    with self.subTest('apply'):
      clients = [
          (b'cid0', fedjax.ClientDataset({'x': jnp.array([2., 4., 6.])}),
           jax.random.PRNGKey(0)),
          (b'cid1', fedjax.ClientDataset({'x': jnp.array([8., 10.])}),
           jax.random.PRNGKey(1)),
      ]
      state, client_diagnostics = algorithm.apply(state, clients)
      npt.assert_allclose(state.params['w'], [0., 1.4425802, 2.8851604])
      npt.assert_allclose(client_diagnostics[b'cid0']['delta_l2_norm'],
                          1.7553135)
      npt.assert_allclose(client_diagnostics[b'cid1']['delta_l2_norm'],
                          0.48310122)


if __name__ == '__main__':
  absltest.main()
