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
"""Tests for util."""

from absl.testing import absltest
from fedjax.core import util
import jax
import jax.numpy as jnp
import numpy.testing as npt

jax.config.update('jax_debug_nans', True)


class UtilsTest(absltest.TestCase):

  def test_safe_div(self):
    # Safe division by zero.
    npt.assert_array_equal(
        util.safe_div(jnp.array([1, 2, 3]), jnp.array([0, 1, 2])), [0, 2, 1.5])
    # Safe gradient when division by zero.
    grad = jax.grad(lambda xy: jnp.sum(util.safe_div(xy[0], xy[1])))
    npt.assert_array_equal(
        grad(jnp.array([[1, 2, 3], [0, 1, 2]], dtype=jnp.float32)),
        [[0, 1, 0.5], [0, -2, -3 / 4]])


if __name__ == '__main__':
  absltest.main()
