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
"""Tests for fedjax.experimental.examples.fed_avg_basic."""
from absl.testing import absltest

from fedjax.experimental import client_datasets
from fedjax.experimental import optimizers
from fedjax.experimental.examples import fed_avg_basic

import jax
import jax.numpy as jnp
import numpy.testing as npt


class FedAvgBasicTest(absltest.TestCase):

  def test_federated_averaging(self):
    grad_fn = lambda p, b, r: jax.tree_util.tree_map(lambda l: l * 0.5, p)
    client_optimizer = optimizers.sgd(learning_rate=1.0)
    server_optimizer = optimizers.sgd(learning_rate=1.0)
    client_dataset_hparams = client_datasets.ShuffleRepeatBatchHParams(
        batch_size=2, num_epochs=1, seed=0)
    fed_avg = fed_avg_basic.federated_averaging(grad_fn, client_optimizer,
                                                server_optimizer,
                                                client_dataset_hparams)
    with self.subTest('init'):
      state = fed_avg.init({'w': jnp.array([0., 2., 4.])})
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
      state, client_diagnostics = fed_avg.apply(state, clients)
      npt.assert_array_almost_equal(state.params['w'], [0., 0.666667, 1.333333])
      npt.assert_equal(
          client_diagnostics, {
              b'cid0': [
                  {'grad_l2_norm': jnp.array(5.)},
                  {'grad_l2_norm': jnp.array(1.25)}
              ],
              b'cid1': [
                  {'grad_l2_norm': jnp.array(5.)}
              ]
          })


if __name__ == '__main__':
  absltest.main()
