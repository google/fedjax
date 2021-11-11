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

import jax
import jax.numpy as jnp
import numpy.testing as npt

from fedjax.aggregators import walsh_hadamard


def random_input(n):
  return jnp.array(jax.random.normal(jax.random.PRNGKey(0), shape=[n]))


class WalshHadamardTest(absltest.TestCase):

  def test_naive_walsh_hadamard_transform(self):
    with self.subTest('size 1'):
      x = random_input(1)
      y = walsh_hadamard.naive_walsh_hadamard_transform(x)
      self.assertEqual(x.shape, y.shape)
      self.assertEqual(x.dtype, y.dtype)
      npt.assert_array_equal(x, y)

    with self.subTest('size 2'):
      x = random_input(2)
      y = walsh_hadamard.naive_walsh_hadamard_transform(x)
      self.assertEqual(x.shape, y.shape)
      self.assertEqual(x.dtype, y.dtype)
      npt.assert_array_equal([x[0] + x[1], x[0] - x[1]], y)

  def test_fast_walsh_hadamard_transform(self):
    # Use a smaller than default small_n so that we don't run out of memory
    # when testing on GPU.
    small_n = 2**5
    for m in [0, 4, 5, 6, 7]:
      x = random_input(2**m)
      y_naive = walsh_hadamard.naive_walsh_hadamard_transform(x)
      with self.subTest(f'top_down, size {2**m}'):
        y = walsh_hadamard.top_down_fast_walsh_hadamard_transform(x, small_n)
        self.assertEqual(x.shape, y.shape)
        self.assertEqual(x.dtype, y.dtype)
        npt.assert_allclose(y_naive, y, rtol=1e-2)
      with self.subTest(f'bottom_up, size {2**m}'):
        y = walsh_hadamard.bottom_up_fast_walsh_hadamard_transform(x, small_n)
        self.assertEqual(x.shape, y.shape)
        self.assertEqual(x.dtype, y.dtype)
        npt.assert_allclose(y_naive, y, rtol=1e-2)

  def test_walsh_hadamard_transform(self):
    small_n = 2**3
    medium_n = 2**5
    for m in [0, 2, 3, 4, 5, 6, 7]:
      with self.subTest(f'size {2**m}'):
        x = random_input(2**m)
        y_naive = walsh_hadamard.naive_walsh_hadamard_transform(x)
        y = walsh_hadamard.walsh_hadamard_transform(x, small_n, medium_n)
        self.assertEqual(x.shape, y.shape)
        self.assertEqual(x.dtype, y.dtype)
        npt.assert_allclose(y_naive, y, rtol=1e-2)


if __name__ == '__main__':
  absltest.main()
