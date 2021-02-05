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
"""Tests for fedjax.training.checkpoint."""

import os

from fedjax.core import test_util
from fedjax.training import checkpoint
import tensorflow as tf


class CheckpointTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._root_dir = self.get_temp_dir()

  def _write_file(self, basename):
    path = os.path.join(self._root_dir, basename)
    with tf.io.gfile.GFile(path, mode='w') as f:
      f.write(basename)

  def test_save_checkpoint(self):
    checkpoint.save_checkpoint(self._root_dir, test_util.create_mock_state())

    self.assertEqual(
        tf.io.gfile.listdir(self._root_dir), ['checkpoint_00000000'])

  def test_save_checkpoint_keep(self):
    state = test_util.create_mock_state()

    for i in range(3):
      checkpoint.save_checkpoint(self._root_dir, state, round_num=i, keep=2)

    self.assertCountEqual(
        tf.io.gfile.listdir(self._root_dir),
        ['checkpoint_00000001', 'checkpoint_00000002'])

  def test_get_checkpoint_paths(self):
    self._write_file('checkpoint_00000100')
    self._write_file('checkpoint_00000000')
    self._write_file('checkpoint_00000003')
    # Similarly named files that shouldn't be counted as valid checkpoints.
    self._write_file('checkpoint_haha')
    self._write_file('checkpoint_0ha')
    self._write_file('checkpoint_00000004.tf.ckpt')

    checkpoint_paths = checkpoint._get_checkpoint_paths(
        os.path.join(self._root_dir, 'checkpoint_'))

    basenames = [os.path.basename(cp) for cp in checkpoint_paths]
    self.assertEqual(
        basenames,
        ['checkpoint_00000000', 'checkpoint_00000003', 'checkpoint_00000100'])

  def test_load_latest_checkpoint(self):
    state_1 = test_util.create_mock_state(seed=1)
    state_2 = test_util.create_mock_state(seed=2)
    checkpoint.save_checkpoint(self._root_dir, state_1, round_num=10, keep=2)
    checkpoint.save_checkpoint(self._root_dir, state_2, round_num=3, keep=2)

    latest_state, latest_round_num = checkpoint.load_latest_checkpoint(
        root_dir=self._root_dir)

    self.assertEqual(latest_round_num, 10)
    tf.nest.map_structure(self.assertAllEqual, latest_state, state_1)


if __name__ == '__main__':
  tf.test.main()
