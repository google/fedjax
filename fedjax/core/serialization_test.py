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
"""Tests for fedjax.core.serialization."""

import os

from fedjax.core import serialization
from fedjax.core import test_util
import jax
import tensorflow as tf


class SerializationTest(tf.test.TestCase):

  def test_save_state(self):
    temp_dir = self.get_temp_dir()
    path = os.path.join(temp_dir, 'state')
    state = test_util.create_mock_state()

    serialization.save_state(state, path)

    self.assertTrue(tf.io.gfile.exists(path))

  def test_load_state(self):
    temp_dir = self.get_temp_dir()
    path = os.path.join(temp_dir, 'state')
    init_state = test_util.create_mock_state()
    serialization.save_state(init_state, path)

    state = serialization.load_state(path)

    expected_flat, expected_tree_def = jax.tree_flatten(init_state)
    actual_flat, actual_tree_def = jax.tree_flatten(state)
    for expected_array, actual_array in zip(expected_flat, actual_flat):
      self.assertAllEqual(expected_array, actual_array)
    self.assertEqual(expected_tree_def, actual_tree_def)


if __name__ == '__main__':
  tf.test.main()
