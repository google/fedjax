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
"""Tests for fedjax.datasets.stackoverflow.

This file only tests preprocessing functions.
"""

from absl.testing import absltest
from fedjax.datasets import stackoverflow
import numpy as np
import numpy.testing as npt


class StackoverflowTest(absltest.TestCase):

  def test_preprocess_client(self):
    npt.assert_equal(
        stackoverflow.preprocess_client(
            None, {
                'tokens':
                    np.array([b'this is a test', b'may it not fail'],
                             dtype=np.object),
                'creation_date':
                    np.array([b'1970-01-01', b'2021-04-20']),
                'type':
                    np.array([b'answer', b'question'])
            }), {
                'tokens':
                    np.array([b'this is a test', b'may it not fail'],
                             dtype=np.object),
                'domain_id': [1, 0]
            })

  def test_tokenizer(self):
    tokenizer = stackoverflow.StackoverflowTokenizer(
        vocab=['three', 'four', 'five'])

    examples = {
        'tokens':
            np.array([b'three four five', b'five three four five six'],
                     dtype=np.object)
    }

    with self.subTest('padding and truncation'):
      # Truncation only.
      npt.assert_equal(
          tokenizer.as_preprocess_batch(max_length=2)(examples), {
              'x': [[1, 3], [1, 5]],
              'y': [[3, 4], [5, 3]]
          })
      # Padding and truncation.
      npt.assert_equal(
          tokenizer.as_preprocess_batch(max_length=5)(examples), {
              'x': [[1, 3, 4, 5, 0], [1, 5, 3, 4, 5]],
              'y': [[3, 4, 5, 2, 0], [5, 3, 4, 5, 6]]
          })
      # Padding only.
      npt.assert_equal(
          tokenizer.as_preprocess_batch(max_length=7)(examples), {
              'x': [[1, 3, 4, 5, 0, 0, 0], [1, 5, 3, 4, 5, 6, 0]],
              'y': [[3, 4, 5, 2, 0, 0, 0], [5, 3, 4, 5, 6, 2, 0]]
          })

    with self.subTest('domain_id'):
      npt.assert_equal(
          tokenizer.as_preprocess_batch(max_length=2)({
              **examples, 'domain_id': np.array([1, 0])
          }), {
              'x': [[1, 3], [1, 5]],
              'y': [[3, 4], [5, 3]],
              'domain_id': [1, 0]
          })

    with self.subTest('num_oov_buckets'):
      npt.assert_equal(
          stackoverflow.StackoverflowTokenizer(
              vocab=['hello'],
              num_oov_buckets=3).as_preprocess_batch(max_length=5)(examples), {
                  'x': [[1, 5, 5, 4, 0], [1, 4, 5, 5, 4]],
                  'y': [[5, 5, 4, 2, 0], [4, 5, 5, 4, 5]]
              })

    with self.subTest('unused_default_vocab_size'):
      npt.assert_equal(
          stackoverflow.StackoverflowTokenizer(
              vocab=['three', 'four', 'five', 'six'],
              default_vocab_size=1).as_preprocess_batch(max_length=6)(examples),
          {
              'x': [[1, 3, 4, 5, 0, 0], [1, 5, 3, 4, 5, 6]],
              'y': [[3, 4, 5, 2, 0, 0], [5, 3, 4, 5, 6, 2]]
          })


if __name__ == '__main__':
  absltest.main()
