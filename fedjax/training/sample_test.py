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
"""Tests for fedjax.training.sample."""

from absl.testing import parameterized
from fedjax.training import sample
import tensorflow as tf


class SampleTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': '_int_no_replace',
          'a': 100,
          'replace': False
      }, {
          'testcase_name': '_int_replace',
          'a': 5,
          'replace': True
      }, {
          'testcase_name': '_sequence_no_replace',
          'a': [str(i) for i in range(100)],
          'replace': False
      }, {
          'testcase_name': '_sequence_replace',
          'a': [str(i) for i in range(5)],
          'replace': True
      })
  def test_build_sample_fn_with_random_seed(self, a, replace):
    size = 10
    random_seed = 1
    round_num = 5

    sample_fn_1 = sample.build_sample_fn(
        a, size, replace=replace, random_seed=random_seed)
    sample_1 = sample_fn_1(round_num)

    sample_fn_2 = sample.build_sample_fn(
        a, size, replace=replace, random_seed=random_seed)
    sample_2 = sample_fn_2(round_num)

    self.assertAllEqual(sample_1, sample_2)

  @parameterized.named_parameters(
      {
          'testcase_name': '_int_no_replace',
          'a': 100,
          'replace': False
      }, {
          'testcase_name': '_int_replace',
          'a': 5,
          'replace': True
      }, {
          'testcase_name': '_sequence_no_replace',
          'a': [str(i) for i in range(100)],
          'replace': False
      }, {
          'testcase_name': '_sequence_replace',
          'a': [str(i) for i in range(5)],
          'replace': True
      })
  def test_build_sample_fn_without_random_seed(self, a, replace):
    size = 10
    round_num = 5

    sample_fn_1 = sample.build_sample_fn(a, size, replace=replace)
    sample_1 = sample_fn_1(round_num)

    sample_fn_2 = sample.build_sample_fn(a, size, replace=replace)
    sample_2 = sample_fn_2(round_num)

    self.assertNotAllEqual(sample_1, sample_2)


if __name__ == '__main__':
  tf.test.main()
