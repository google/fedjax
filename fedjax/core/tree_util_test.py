# Copyright 2020 Google LLC
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

from fedjax.core import tree_util
import jax.numpy as jnp
import tensorflow as tf


class TreeUtilTest(tf.test.TestCase):

  def test_tree_broadcast(self):
    pytree = {'x': jnp.array([[0, 0]]), 'y': jnp.array([0])}
    broadcast_pytree = tree_util.tree_broadcast(pytree, axis_size=1)
    self.assertAllEqual(broadcast_pytree['x'], [[[0, 0]]])
    self.assertAllEqual(broadcast_pytree['y'], [[0]])

  def test_tree_stack(self):
    pytrees = [
        {
            'x': jnp.array([[0, 0]]),
            'y': jnp.array([0])
        },
        {
            'x': jnp.array([[1, 1]]),
            'y': jnp.array([1])
        },
        {
            'x': jnp.array([[2, 2]]),
            'y': jnp.array([2])
        },
    ]
    pytree = tree_util.tree_stack(pytrees)
    self.assertAllEqual(pytree['x'], [[[0, 0]], [[1, 1]], [[2, 2]]])
    self.assertAllEqual(pytree['y'], [[0], [1], [2]])

  def test_tree_unstack(self):
    pytree = {
        'x': jnp.array([[[0, 0]], [[1, 1]]]),
        'y': jnp.array([[0], [1]]),
    }
    pytrees = list(tree_util.tree_unstack(pytree, axis_size=2))
    self.assertAllEqual(pytrees[0]['x'], [[0, 0]])
    self.assertAllEqual(pytrees[0]['y'], [0])
    self.assertAllEqual(pytrees[1]['x'], [[1, 1]])
    self.assertAllEqual(pytrees[1]['y'], [1])

  def test_tree_weight(self):
    pytree = {
        'x': jnp.array([[[4, 5]], [[1, 1]]]),
        'y': jnp.array([[3], [1]]),
    }
    weight = 2.0
    weight_pytree = tree_util.tree_weight(pytree, weight)
    self.assertAllEqual(weight_pytree['x'], [[[8.0, 10.0]], [[2.0, 2.0]]])
    self.assertAllEqual(weight_pytree['y'], [[6.0], [2.0]])

  def test_tree_sum(self):
    pytree_1 = {
        'x': jnp.array([[[4, 5]], [[1, 1]]]),
        'y': jnp.array([[3], [1]]),
    }
    pytree_2 = {
        'x': jnp.array([[[2, 3]], [[4, 5]]]),
        'y': jnp.array([[6], [7]]),
    }
    pytree = tree_util.tree_sum(pytree_1, pytree_2)
    self.assertAllEqual(pytree['x'], [[[6, 8]], [[5, 6]]])
    self.assertAllEqual(pytree['y'], [[9], [8]])

  def test_tree_mean(self):
    pytrees = [(0, 1), (2, 3), (4, 5)]
    weights = [6., 7., 8.]
    pytrees_and_weights = zip(pytrees, weights)
    pytree = tree_util.tree_mean(pytrees_and_weights)
    self.assertAllClose(pytree, (2.1904761904761907, 3.1904761904761907))


if __name__ == '__main__':
  tf.test.main()
