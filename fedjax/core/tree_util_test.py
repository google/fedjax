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
"""Tests for fedjax.core.tree_util."""

from absl.testing import absltest

from fedjax.core import tree_util

import jax.numpy as jnp
import numpy.testing as npt


class TreeUtilTest(absltest.TestCase):

  def test_tree_weight(self):
    pytree = {
        'x': jnp.array([[[4, 5]], [[1, 1]]]),
        'y': jnp.array([[3], [1]]),
    }
    weight = 2.0
    weight_pytree = tree_util.tree_weight(pytree, weight)
    npt.assert_array_equal(weight_pytree['x'], [[[8.0, 10.0]], [[2.0, 2.0]]])
    npt.assert_array_equal(weight_pytree['y'], [[6.0], [2.0]])

  def test_tree_sum(self):
    pytree_1 = {
        'x': jnp.array([[[4, 5]], [[1, 1]]]),
        'y': jnp.array([[3], [1]]),
    }
    pytree_2 = {
        'x': jnp.array([[[2, 3]], [[4, 5]]]),
        'y': jnp.array([[6], [7]]),
    }
    pytree = tree_util.tree_sum([pytree_1, pytree_2])
    npt.assert_array_equal(pytree['x'], [[[6, 8]], [[5, 6]]])
    npt.assert_array_equal(pytree['y'], [[9], [8]])

  def test_tree_mean(self):
    pytrees = [(0, 1), (2, 3), (4, 5)]
    weights = [6., 7., 8.]
    pytrees_and_weights = zip(pytrees, weights)
    pytree = tree_util.tree_mean(pytrees_and_weights)
    npt.assert_array_almost_equal(pytree,
                                  (2.1904761904761907, 3.1904761904761907))

  def test_tree_clip_by_global_norm(self):
    pytree = {
        'x': jnp.array([[[4, 5]], [[1, 1]]]),
        'y': jnp.array([[3], [1]]),
    }
    max_norm = 3.640055  # 0.5 * tree_l2_norm(pytree)
    clipped_pytree = tree_util.tree_clip_by_global_norm(pytree, max_norm)
    npt.assert_array_almost_equal(clipped_pytree['x'],
                                  [[[2, 2.5]], [[0.5, 0.5]]])
    npt.assert_array_almost_equal(clipped_pytree['y'], [[1.5], [0.5]])


if __name__ == '__main__':
  absltest.main()
