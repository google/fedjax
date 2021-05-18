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
"""Tests for fedjax.core.regularizers."""

from absl.testing import absltest

from fedjax.core import regularizers

import jax
import jax.numpy as jnp


class RegularizersTest(absltest.TestCase):

  def test_l2_regularizer(self):
    params = {
        'linear0': {
            'w': jnp.array([1., 2., 3.])
        },
        'linear1': {
            'w': jnp.array([4., 5., 6.]),
            'b': jnp.array([1., 1., 1.])
        }
    }

    with self.subTest('weight'):
      regularizer = regularizers.l2_regularizer(weight=0.2)
      self.assertAlmostEqual(regularizer(params), 0.2 * 94, places=5)

    with self.subTest('center_params'):
      center_params = jax.tree_util.tree_map(jnp.ones_like, params)
      regularizer = regularizers.l2_regularizer(
          weight=0.2, center_params=center_params)
      self.assertAlmostEqual(regularizer(params), 0.2 * 55)

    with self.subTest('params_weights'):
      params_weights = jax.tree_util.tree_map(lambda l: jnp.ones_like(l) * 2,
                                              params)
      regularizer = regularizers.l2_regularizer(
          weight=0.2, params_weights=params_weights)
      self.assertAlmostEqual(regularizer(params), 0.2 * 188, places=5)


if __name__ == '__main__':
  absltest.main()
