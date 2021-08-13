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
"""Test for data_to_sqlite dialogue normalization."""

from absl.testing import absltest
from absl.testing import parameterized
from fedjax.datasets.scripts.cornell_movie_dialogs import data_to_sqlite


class TxtToSQLiteTest(parameterized.TestCase):
  @parameterized.named_parameters(
      {
          'testcase_name': 'test hyphenated',
          'input_string': 'trigger-happy?',
          'desired': 'trigger-happy',
      }, {
          'testcase_name': 'test sentence',
          'input_string': 'Thank God! If I had to hear one more story...\n',
          'desired': 'thank god if i had to hear one more story',
      }, {
          'testcase_name': 'test decimal',
          'input_string': 'Pi is 3.14 and pi day is in 10      months.',
          'desired': 'pi is <NUMERIC> and pi day is in <NUMERIC> months',
      }, {
          'testcase_name': 'test dollar',
          'input_string': 'I have 50,000.00 dollars in my account right now.\n',
          'desired': 'i have <NUMERIC> dollars in my account right now',
      }, {
          'testcase_name': 'test hyphen',
          'input_string': 'This is a double--hyphen! And this is a single-one.',
          'desired': 'this is a double hyphen and this is a single-one',
      }, {
          'testcase_name': 'test ellipses',
          'input_string': 'I..do...not......know.....\n',
          'desired': 'i do not know',
      }, {
          'testcase_name':
              'test html tags',
          'input_string':
              '<u>These are</u> underlining HTML and <b>bold</b> HTML tags.',
          'desired':
              'these are underlining html and bold html tags',
      })

  def test_normalize_dialogue_string(self, input_string, desired):
    self.assertEqual(
        desired,
        data_to_sqlite.normalize_dialogue_string(input_string))


if __name__ == '__main__':
  absltest.main()
