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
"""Tests for fedjax.experimental.federated_data."""

from fedjax.experimental import federated_data
from absl.testing import absltest


class RepeatableIteratorTest(absltest.TestCase):

  def test_two_passes(self):
    # range(5) is repeatable, iter(range(5)) is not.
    it = federated_data.RepeatableIterator(iter(range(5)))
    self.assertListEqual(list(it), [0, 1, 2, 3, 4])
    self.assertListEqual(list(it), [0, 1, 2, 3, 4])

  def test_no_copy_cases(self):
    for values in [[1, 2, 3], (1, 2, 3), '123', b'123', {'1': 2, '3': 4}]:
      it = federated_data.RepeatableIterator(values)
      self.assertIs(values, it._buf)
      self.assertListEqual(list(it), list(values))
      self.assertListEqual(list(it), list(values))


if __name__ == '__main__':
  absltest.main()
