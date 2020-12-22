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
"""Tests for fedjax.datasets.emnist."""

from fedjax.datasets import emnist
import tensorflow as tf


class EmnistDataTest(tf.test.TestCase):

  def test_flip_and_expand(self):
    dataset = tf.data.Dataset.from_tensor_slices({
        'pixels': [[[1.0, 0.9], [0.8, 1.0]]],
        'label': [2]
    })

    output = next(emnist.flip_and_expand(dataset).as_numpy_iterator())

    self.assertAllClose(output['x'], [[[0.], [0.1]], [[0.2], [0.]]])
    self.assertAllEqual(output['y'], 2)


if __name__ == '__main__':
  tf.test.main()
