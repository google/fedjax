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
"""Tests for fedjax.datasets.emnist.

This file only tests preprocessing functions.
"""

from absl.testing import absltest
from fedjax.datasets import emnist
import numpy as np
import numpy.testing as npt


class EmnistTest(absltest.TestCase):

  def test_domain_id(self):
    with self.subTest('invalid'):
      with self.assertRaisesRegex(ValueError, 'Invalid client_id'):
        emnist.domain_id(b'not_emnist')

    with self.subTest('domain 0'):
      for client_id in [b'f2100_xx', b'f2222_yy', b'f2599_zz']:
        self.assertEqual(emnist.domain_id(client_id), 0)
        self.assertEqual(emnist.domain_id(b'A' * 16 + b':' + client_id), 0)

    with self.subTest('domain 1'):
      for client_id in [b'f0000_xx', b'f2099_yy', b'f2600_zz', b'f9999_99']:
        self.assertEqual(emnist.domain_id(client_id), 1)
        self.assertEqual(emnist.domain_id(b'A' * 16 + b':' + client_id), 1)

  def test_preprocess(self):
    pixels = np.random.uniform(size=[128, 28, 28])
    label = np.random.randint(10, size=[128])
    npt.assert_equal(
        emnist.preprocess_client(b'f0000_xx', {
            'pixels': pixels,
            'label': label
        }), {
            'pixels': pixels,
            'label': label,
            'domain_id': np.ones_like(label)
        })

    domain_id = np.random.randint(2, size=[128])
    npt.assert_equal(
        emnist.preprocess_batch({
            'pixels': pixels,
            'label': label,
            'domain_id': domain_id
        }), {
            'x': 1 - pixels[..., np.newaxis],
            'y': label,
            'domain_id': domain_id
        })


if __name__ == '__main__':
  absltest.main()
