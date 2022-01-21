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
"""Tests for walsh_hadamard."""

from absl.testing import absltest
from fedjax.aggregators import walsh_hadamard
import jax
import jax.numpy as jnp
import numpy.testing as npt


class WalshHadamardTest(absltest.TestCase):

  def test_walsh_hadamard_transform(self):
    # TODO(wuke): Change bfloat16_3x to high once the PyPI JAX release catches
    # up.
    for precision in ['fastest', 'bfloat16_3x', 'highest']:
      for seed in range(5):
        n = 2**10
        x = jnp.array(jax.random.normal(jax.random.PRNGKey(seed), shape=[n]))
        expect = jnp.dot(
            walsh_hadamard.hadamard_matrix(n, x.dtype), x, precision=precision)
        for m in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
          small_n = 2**m
          with self.subTest(
              f'precision={precision}, seed={seed}, small_n={small_n}'):
            y = walsh_hadamard.walsh_hadamard_transform(x, small_n)
            self.assertEqual(x.shape, y.shape)
            self.assertEqual(x.dtype, y.dtype)
            npt.assert_allclose(y, expect, rtol=1e-4, atol=1e-4)

  def test_structured_rotation(self):
    n = 10
    m = 5
    x = jnp.array(jax.random.normal(jax.random.PRNGKey(100), shape=[m, n]))
    rng = jax.random.PRNGKey(10)
    y, x_shape = walsh_hadamard.structured_rotation(x, rng)
    self.assertEqual(x.shape, x_shape)
    z = walsh_hadamard.inverse_structured_rotation(y, rng, x_shape)
    npt.assert_allclose(z, x, rtol=1e-4, atol=1e-4)

  def test_structured_rotation_pytree(self):
    params = {
        'a': jnp.array([[1.0, 0.0, 0.0], [1.0, 2.0, 3.0]]),
        'b': jnp.array([[1.0, 0.0], [1.0, 2.0]])
    }
    rng = jax.random.PRNGKey(10)
    y, shapes = walsh_hadamard.structured_rotation_pytree(params, rng)
    z = walsh_hadamard.inverse_structured_rotation_pytree(y, rng, shapes)
    npt.assert_allclose(z['a'], params['a'], rtol=1e-4, atol=1e-4)
    npt.assert_allclose(z['b'], params['b'], rtol=1e-4, atol=1e-4)


  if __name__ == '__main__':
    absltest.main()
