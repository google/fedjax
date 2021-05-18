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
"""Tests for fedjax.core.serialization."""

from absl.testing import absltest
from fedjax.core import serialization
import numpy as np
import numpy.testing as npt


class SerializationTest(absltest.TestCase):

  def test_dict(self):
    original = {
        'int32':
            np.arange(4, dtype=np.int32).reshape([2, 2]),
        'float64':
            -np.arange(4, dtype=np.float64).reshape([1, 4]),
        'bytes':
            np.array([b'a', b'bc', b'def'], dtype=np.object).reshape([3, 1]),
    }
    output = serialization.msgpack_deserialize(
        serialization.msgpack_serialize(original))
    self.assertCountEqual(output, original)
    self.assertEqual(output['int32'].dtype, np.int32)
    npt.assert_array_equal(output['int32'], original['int32'])
    self.assertEqual(output['float64'].dtype, np.float64)
    npt.assert_array_equal(output['float64'], original['float64'])
    self.assertEqual(output['bytes'].dtype, np.object)
    npt.assert_array_equal(output['bytes'], original['bytes'])

  def test_nested_list(self):
    original = [
        np.arange(4, dtype=np.int32).reshape([2, 2]),
        [
            -np.arange(4, dtype=np.float64).reshape([1, 4]),
            [
                np.array([b'a', b'bc', b'def'],
                         dtype=np.object).reshape([3, 1]), []
            ]
        ]
    ]
    output = serialization.msgpack_deserialize(
        serialization.msgpack_serialize(original))
    int32_array, rest = output
    self.assertEqual(int32_array.dtype, np.int32)
    npt.assert_array_equal(int32_array, original[0])
    float64_array, rest = rest
    self.assertEqual(float64_array.dtype, np.float64)
    npt.assert_array_equal(float64_array, original[1][0])
    bytes_array, rest = rest
    self.assertEqual(bytes_array.dtype, np.object)
    npt.assert_array_equal(bytes_array, original[1][1][0])
    self.assertEqual(rest, [])


if __name__ == '__main__':
  absltest.main()
