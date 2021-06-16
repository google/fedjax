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
"""Tests for fedjax.datasets.shakespeare.

This file only tests preprocessing functions.
"""

from absl.testing import absltest
from fedjax.datasets import shakespeare
import numpy as np
import numpy.testing as npt


class ShakespeareTest(absltest.TestCase):

  def test_preprocess_client(self):
    examples = shakespeare.preprocess_client(
        None, {'snippets': np.array([b'hello', b'hi there'], dtype=np.object)},
        sequence_length=10)
    npt.assert_equal(
        examples, {
            'x': [[1, 4, 67, 5, 5, 26, 2, 1, 4, 68],
                  [16, 7, 4, 67, 48, 67, 0, 0, 0, 0]],
            'y': [[4, 67, 5, 5, 26, 2, 1, 4, 68, 16],
                  [7, 4, 67, 48, 67, 2, 0, 0, 0, 0]],
        })
    npt.assert_equal(examples['x'].dtype, np.int32)
    npt.assert_equal(examples['y'].dtype, np.int32)


if __name__ == '__main__':
  absltest.main()
