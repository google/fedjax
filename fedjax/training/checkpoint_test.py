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

from absl.testing import absltest

from fedjax.training import checkpoint
import jax.numpy as jnp
import numpy.testing as npt


class CheckpointTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._root_dir = self.create_tempdir()

  def _write_file(self, basename):
    path = os.path.join(self._root_dir, basename)
    with open(path, 'w') as f:
      f.write(basename)

  def test_save_checkpoint(self):
    checkpoint.save_checkpoint(self._root_dir, {'w': jnp.ones(10)})

    self.assertEqual(os.listdir(self._root_dir), ['checkpoint_00000000'])

  def test_save_checkpoint_keep(self):
    state = {'w': jnp.ones(10)}

    for i in range(3):
      checkpoint.save_checkpoint(self._root_dir, state, round_num=i, keep=2)

    self.assertCountEqual(
        os.listdir(self._root_dir),
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
    state_1 = {'w': jnp.ones(10)}
    state_2 = {'w': jnp.ones(10) * 2}
    checkpoint.save_checkpoint(self._root_dir, state_1, round_num=1, keep=2)
    checkpoint.save_checkpoint(self._root_dir, state_2, round_num=2, keep=2)

    latest_state, latest_round_num = checkpoint.load_latest_checkpoint(
        root_dir=self._root_dir)

    self.assertEqual(latest_round_num, 2)
    npt.assert_array_equal(latest_state['w'], state_2['w'])


if __name__ == '__main__':
  absltest.main()
