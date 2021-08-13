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
# limitations under the License.Let's use the full copyright notice
"""Test for data_to_sqlite normalization."""

from absl.testing import absltest
from absl.testing import parameterized
from fedjax.datasets.scripts.sent140 import data_to_sqlite


class DataToSQLiteTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'test url and username',
          'input_string': 'Thank @user1 https://en.wikipedia.org more story...',
          'desired': 'thank USERNAME URL more story',
      }, {
          'testcase_name': 'test numerics',
          'input_string': 'Pi 3.14, 10 months, 50,000.00 and 7-10.',
          'desired': 'pi NUMERIC NUMERIC months NUMERIC and NUMERIC',
      }, {
          'testcase_name': 'test hyphen',
          'input_string': 'This is a double--hyphen! And this is a single-one.',
          'desired': 'this is a double hyphen and this is a single-one',
      }, {
          'testcase_name': 'test punctuation',
          'input_string': 'I.do{not},know....that*',
          'desired': 'i do not know that',
      })
  def test_normalize_dialogue_string(self, input_string, desired):
    self.assertEqual(desired, data_to_sqlite.normalize_tweet(input_string))


if __name__ == '__main__':
  absltest.main()
