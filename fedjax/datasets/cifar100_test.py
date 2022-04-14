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
"""Tests for fedjax.datasets.cifar100.

This file only tests preprocessing functions.
"""

from absl.testing import absltest
from fedjax.datasets import cifar100
import numpy as np
import numpy.testing as npt


class Cifar100Test(absltest.TestCase):

  def test_preprocess_client(self):
    image = np.random.randint(256, size=[4, 32, 32, 3]).astype(np.uint8)
    coarse_label = np.random.randint(20, size=[4]).astype(np.int64)
    label = np.random.randint(100, size=[4]).astype(np.int64)
    npt.assert_equal(
        cifar100.preprocess_client(None, {
            'image': image,
            'coarse_label': coarse_label,
            'label': label
        }), {
            'x': image,
            'y': label
        })

  def test_preprocess_image(self):
    # 4 images, each with one side being 1 after processing.
    image = np.zeros([4, 32, 32, 3], dtype=np.uint8)
    # [125, 123, 114] becomes nearly 0 after processing.
    image[:, :, :, :] = [125, 123, 114]
    # [177, 174, 165] becomes nearly 1 after processing.
    image[0, 0, :, :] = [177, 174, 165]
    image[1, -1, :, :] = [177, 174, 165]
    image[2, :, 0, :] = [177, 174, 165]
    image[3, :, -1, :] = [177, 174, 165]

    with self.subTest('is_train=False'):
      processed = cifar100.preprocess_image(image, is_train=False)
      self.assertEqual(processed.dtype, np.float32)
      ref = np.zeros([4, 32, 32, 3], dtype=np.float32)
      ref[0, 0, :, :] = 1
      ref[1, -1, :, :] = 1
      ref[2, :, 0, :] = 1
      ref[3, :, -1, :] = 1
      npt.assert_allclose(processed, ref, rtol=0.01, atol=0.01)

    with self.subTest('is_train=True'):
      # Expected crop start offsets: (2, 6); flip: yes.
      np.random.seed(4)
      processed = cifar100.preprocess_image(image, is_train=True)
      self.assertEqual(processed.dtype, np.float32)
      ref = np.zeros([4, 32, 32, 3], dtype=np.float32)
      padding = [-2.4290657, -2.4182549, -2.221393]
      ref[:, :2, :, :] = padding
      ref[:, :, :2, :] = padding
      ref[0, 2, 2:, :] = 1
      ref[3, 2:, 2, :] = 1
      npt.assert_allclose(processed, ref, rtol=0.01, atol=0.01)

  def test_preprocess_batch(self):
    np.random.seed(0)
    examples = {
        'x': np.random.randint(256, size=[4, 32, 32, 3]).astype(np.uint8),
        'y': np.random.randint(100, size=[4]).astype(np.int32)
    }

    train_processed = cifar100.preprocess_batch(examples, is_train=True)
    self.assertCountEqual(train_processed, ['x', 'y'])
    self.assertIsNot(train_processed['x'], examples['x'])
    self.assertEqual(train_processed['x'].dtype, np.float32)
    self.assertEqual(train_processed['x'].shape, (4, 32, 32, 3))
    self.assertIs(train_processed['y'], examples['y'])

    eval_processed = cifar100.preprocess_batch(examples, is_train=False)
    self.assertCountEqual(eval_processed, ['x', 'y'])
    self.assertIsNot(eval_processed['x'], examples['x'])
    self.assertEqual(eval_processed['x'].dtype, np.float32)
    self.assertEqual(eval_processed['x'].shape, (4, 32, 32, 3))
    self.assertIs(eval_processed['y'], examples['y'])

    self.assertTrue(np.any(train_processed['x'] != eval_processed['x']))

  def test_preprocess_image_tff(self):
    image = np.random.RandomState(0).choice(
        256, size=(2, 32, 32, 3)).astype(np.uint8)
    crop_height = 24
    crop_width = 24

    with self.subTest('distort=False'):
      processed = cifar100.preprocess_image_tff(
          image, crop_height, crop_width, distort=False)
      self.assertEqual(processed.dtype, np.float32)
      self.assertTupleEqual(processed.shape, (2, crop_height, crop_width, 3))

    with self.subTest('distort=True'):
      processed = cifar100.preprocess_image_tff(
          image, crop_height, crop_width, distort=True)
      self.assertEqual(processed.dtype, np.float32)
      self.assertTupleEqual(processed.shape, (2, crop_height, crop_width, 3))

  def test_preprocess_image_tff_invalid_crop(self):
    image = np.zeros((2, 32, 32, 3)).astype(np.uint8)
    with self.assertRaisesRegex(
        ValueError, 'The crop_height and crop_width must be between 1 and 32'):
      cifar100.preprocess_image_tff(
          image, crop_height=33, crop_width=-1, distort=False)


if __name__ == '__main__':
  absltest.main()
