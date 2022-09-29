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
"""Tests for fedjax.core.optimizers."""

from absl.testing import absltest

from fedjax.core import optimizers

import haiku as hk
import jax
import jax.numpy as jnp
import numpy.testing as npt


class OptimizersTest(absltest.TestCase):

  def test_ignore_grads_haiku(self):
    params = hk.data_structures.to_immutable_dict({
        'linear_1': {
            'w': jnp.array([1., 1., 1.])
        },
        'linear_2': {
            'w': jnp.array([2., 2., 2.]),
            'b': jnp.array([3., 3., 3.])
        }
    })
    grads = jax.tree_util.tree_map(lambda _: 0.5, params)
    ignore_optimizer = optimizers.ignore_grads_haiku(
        optimizer=optimizers.sgd(learning_rate=1.0),
        non_trainable_names=[('linear_1', 'w'), ('linear_2', 'b')])

    opt_state = ignore_optimizer.init(params)
    opt_state, updated_params = ignore_optimizer.apply(grads, opt_state, params)

    jax.tree_util.tree_map(
        npt.assert_array_equal, updated_params,
        hk.data_structures.to_immutable_dict({
            'linear_1': {
                'w': jnp.array([1., 1., 1.])
            },
            'linear_2': {
                'w': jnp.array([1.5, 1.5, 1.5]),
                'b': jnp.array([3., 3., 3.])
            }
        }))


if __name__ == '__main__':
  absltest.main()
