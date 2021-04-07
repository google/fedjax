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
"""Tests for fedjax.experimental.client_datasets."""

import itertools

from absl.testing import absltest
from fedjax.experimental import client_datasets
import numpy as np
import numpy.testing as npt

# pylint: disable=g-long-lambda


class ExamplesTest(absltest.TestCase):

  def test_num_examples(self):
    with self.subTest('consistent rows'):
      self.assertEqual(
          client_datasets.num_examples({'pixels': np.zeros([10, 20, 30])}), 10)
    with self.subTest('inconsistent rows'):
      with self.assertRaises(ValueError):
        client_datasets.num_examples({
            'a': np.zeros([10, 20]),
            'b': np.zeros([20, 10])
        })

  def test_slice_examples(self):
    examples = {'a': np.arange(10), 'b': np.arange(20).reshape([10, 2])}

    with self.subTest('Slicing like [3:]'):
      sliced = client_datasets.slice_examples(examples, slice(3))
      self.assertCountEqual(sliced, ['a', 'b'])
      npt.assert_array_equal(sliced['a'], [0, 1, 2])
      npt.assert_array_equal(sliced['b'], [[0, 1], [2, 3], [4, 5]])

    with self.subTest('Slicing like [3:6]'):
      sliced = client_datasets.slice_examples(examples, slice(3, 6))
      self.assertCountEqual(sliced, ['a', 'b'])
      npt.assert_array_equal(sliced['a'], [3, 4, 5])
      npt.assert_array_equal(sliced['b'], [[6, 7], [8, 9], [10, 11]])

    with self.subTest('Slicing like [3:6:2]'):
      sliced = client_datasets.slice_examples(examples, slice(3, 6, 2))
      self.assertCountEqual(sliced, ['a', 'b'])
      npt.assert_array_equal(sliced['a'], [3, 5])
      npt.assert_array_equal(sliced['b'], [[6, 7], [10, 11]])

  def test_pad_examples(self):
    examples = {'a': np.arange(5), 'b': np.arange(10).reshape([5, 2])}
    with self.subTest('mask_key already used'):
      with self.assertRaises(ValueError):
        client_datasets.pad_examples(examples, size=8, mask_key='a')
    with self.subTest('size too small'):
      with self.assertRaises(ValueError):
        client_datasets.pad_examples(examples, size=4, mask_key='mask')
    with self.subTest('no padding needed'):
      self.assertIs(client_datasets.pad_examples(examples, 5, 'mask'), examples)
    with self.subTest('padding needed'):
      padded = client_datasets.pad_examples(examples, 6, 'mask')
      self.assertCountEqual(padded, ['a', 'b', 'mask'])
      npt.assert_array_equal(padded['a'], [0, 1, 2, 3, 4, 0])
      npt.assert_array_equal(padded['b'],
                             [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [0, 0]])
      npt.assert_array_equal(padded['mask'],
                             [True, True, True, True, True, False])


class PreprocessorTest(absltest.TestCase):

  def test_preprocessor(self):
    preprocessor = client_datasets.Preprocessor([
        # Flattens `pixels`.
        lambda x: {
            **x, 'pixels': x['pixels'].reshape([-1, 28 * 28])
        },
        # Introduce `binary_label`.
        lambda x: {
            **x, 'binary_label': x['label'] % 2
        },
    ])
    fake_emnist = {
        'pixels': np.random.uniform(size=(10, 28, 28)),
        'label': np.random.randint(10, size=(10,))
    }

    with self.subTest('2 step preprocessing'):
      result = preprocessor(fake_emnist)
      self.assertCountEqual(result, ['pixels', 'label', 'binary_label'])
      npt.assert_array_equal(result['pixels'],
                             fake_emnist['pixels'].reshape([-1, 28 * 28]))
      self.assertIs(result['label'], fake_emnist['label'])
      npt.assert_array_equal(result['binary_label'], fake_emnist['label'] % 2)

  def test_append(self):
    preprocessor = client_datasets.Preprocessor([
        # Flattens `pixels`.
        lambda x: {
            **x, 'pixels': x['pixels'].reshape([-1, 28 * 28])
        },
        # Introduce `binary_label`.
        lambda x: {
            **x, 'binary_label': x['label'] % 2
        },
    ])
    new_preprocessor = preprocessor.append(lambda x: {
        **x, 'sum_pixels': np.sum(x['pixels'], axis=1)
    })
    self.assertIsNot(new_preprocessor, preprocessor)
    fake_emnist = {
        'pixels': np.random.uniform(size=(10, 28, 28)),
        'label': np.random.randint(10, size=(10,))
    }
    result = new_preprocessor(fake_emnist)
    self.assertCountEqual(result,
                          ['pixels', 'label', 'binary_label', 'sum_pixels'])
    npt.assert_array_equal(result['pixels'],
                           fake_emnist['pixels'].reshape([-1, 28 * 28]))
    self.assertIs(result['label'], fake_emnist['label'])
    npt.assert_array_equal(result['binary_label'], fake_emnist['label'] % 2)
    npt.assert_allclose(result['sum_pixels'],
                        np.sum(fake_emnist['pixels'], axis=(1, 2)))


class ClientDatasetTest(absltest.TestCase):

  def test_inconsistent_shape(self):
    with self.assertRaises(ValueError):
      client_datasets.ClientDataset({'a': np.arange(5), 'b': np.arange(4)})

  def test_len(self):
    examples = {'a': np.arange(10)}
    self.assertLen(client_datasets.ClientDataset(examples), 10)

  def test_batch(self):
    d = client_datasets.ClientDataset(
        {
            'a': np.arange(5),
            'b': np.arange(10).reshape([5, 2])
        }, client_datasets.Preprocessor([lambda x: {
            **x, 'a': 2 * x['a']
        }]))
    view = d.batch(3)
    # `view` should be repeatedly iterable.
    for _ in range(2):
      batches = list(view)
      self.assertLen(batches, 2)
      batch_0, batch_1 = batches
      self.assertCountEqual(batch_0, ['a', 'b'])
      npt.assert_array_equal(batch_0['a'], [0, 2, 4])
      npt.assert_array_equal(batch_0['b'], [[0, 1], [2, 3], [4, 5]])
      self.assertCountEqual(batch_1, ['a', 'b'])
      npt.assert_array_equal(batch_1['a'], [6, 8])
      npt.assert_array_equal(batch_1['b'], [[6, 7], [8, 9]])

  def test_pick_final_batch_size(self):
    self.assertEqual(client_datasets._pick_final_batch_size(16, 8, 4), 8)
    self.assertEqual(client_datasets._pick_final_batch_size(10, 8, 1), 8)
    self.assertEqual(client_datasets._pick_final_batch_size(10, 8, 2), 4)
    self.assertEqual(client_datasets._pick_final_batch_size(10, 8, 3), 2)
    self.assertEqual(client_datasets._pick_final_batch_size(10, 8, 4), 2)
    self.assertEqual(client_datasets._pick_final_batch_size(10, 8, 5), 2)
    self.assertEqual(client_datasets._pick_final_batch_size(9, 8, 1), 8)
    self.assertEqual(client_datasets._pick_final_batch_size(9, 8, 2), 4)
    self.assertEqual(client_datasets._pick_final_batch_size(9, 8, 3), 2)
    self.assertEqual(client_datasets._pick_final_batch_size(9, 8, 4), 1)

  def test_batch_padding(self):
    d = client_datasets.ClientDataset(
        {
            'a': np.arange(5),
            'b': np.arange(10).reshape([5, 2])
        }, client_datasets.Preprocessor([lambda x: {
            **x, 'a': 2 * x['a']
        }]))
    view = d.batch(3, num_batch_size_buckets=1, mask_key='M')
    # `view` should be repeatedly iterable.
    for _ in range(2):
      batches = list(view)
      self.assertLen(batches, 2)
      batch_0, batch_1 = batches
      self.assertCountEqual(batch_0, ['a', 'b'])
      npt.assert_array_equal(batch_0['a'], [0, 2, 4])
      npt.assert_array_equal(batch_0['b'], [[0, 1], [2, 3], [4, 5]])
      self.assertCountEqual(batch_1, ['a', 'b', 'M'])
      npt.assert_array_equal(batch_1['a'], [6, 8, 0])
      npt.assert_array_equal(batch_1['b'], [[6, 7], [8, 9], [0, 0]])
      npt.assert_array_equal(batch_1['M'], [True, True, False])

  def test_shuffle_repeat_batch(self):
    d = client_datasets.ClientDataset(
        {
            'a': np.arange(5),
            'b': np.arange(10).reshape([5, 2])
        }, client_datasets.Preprocessor([lambda x: {
            **x, 'a': 2 * x['a']
        }]))
    # Number of batches under different num_epochs/num_steps combinations.
    with self.subTest('repeating'):
      self.assertLen(list(d.shuffle_repeat_batch(5, num_epochs=1)), 1)
      self.assertLen(list(d.shuffle_repeat_batch(3, num_epochs=1)), 2)
      self.assertLen(list(d.shuffle_repeat_batch(1, num_epochs=1)), 5)
      self.assertLen(list(d.shuffle_repeat_batch(5, num_steps=4)), 4)
      self.assertLen(list(d.shuffle_repeat_batch(3, num_steps=4)), 4)
      self.assertLen(list(d.shuffle_repeat_batch(1, num_steps=4)), 4)
      self.assertLen(
          list(d.shuffle_repeat_batch(5, num_epochs=1, num_steps=4)), 1)
      self.assertLen(
          list(d.shuffle_repeat_batch(3, num_epochs=1, num_steps=4)), 2)
      self.assertLen(
          list(d.shuffle_repeat_batch(1, num_epochs=1, num_steps=4)), 4)
      # 100 is as good as forever.
      self.assertLen(
          list(itertools.islice(d.shuffle_repeat_batch(3), 100)), 100)

    # Check proper shuffling.
    with self.subTest('shuffling'):
      view = d.shuffle_repeat_batch(3, num_steps=4, seed=1)
      # `view` should be repeatedly iterable.
      for _ in range(2):
        batches = list(view)
        self.assertLen(batches, 4)
        self.assertCountEqual(batches[0], ['a', 'b'])
        npt.assert_array_equal(batches[0]['a'], [4, 2, 8])
        npt.assert_array_equal(batches[0]['b'], [[4, 5], [2, 3], [8, 9]])
        self.assertCountEqual(batches[1], ['a', 'b'])
        npt.assert_array_equal(batches[1]['a'], [0, 6, 4])
        npt.assert_array_equal(batches[1]['b'], [[0, 1], [6, 7], [4, 5]])
        self.assertCountEqual(batches[2], ['a', 'b'])
        npt.assert_array_equal(batches[2]['a'], [8, 6, 0])
        npt.assert_array_equal(batches[2]['b'], [[8, 9], [6, 7], [0, 1]])
        self.assertCountEqual(batches[3], ['a', 'b'])
        npt.assert_array_equal(batches[3]['a'], [2, 6, 0])
        npt.assert_array_equal(batches[3]['b'], [[2, 3], [6, 7], [0, 1]])

  def test_slice(self):
    d = client_datasets.ClientDataset(
        {
            'a': np.arange(5),
            'b': np.arange(10).reshape([5, 2])
        }, client_datasets.Preprocessor([lambda x: {
            **x, 'a': 2 * x['a']
        }]))

    with self.subTest('slice [:3]'):
      sliced = d[:3]
      batch = next(iter(sliced.batch(3)))
      self.assertCountEqual(batch, ['a', 'b'])
      npt.assert_array_equal(batch['a'], [0, 2, 4])
      npt.assert_array_equal(batch['b'], [[0, 1], [2, 3], [4, 5]])

    with self.subTest('slice [-3:]'):
      sliced = d[-3:]
      batch = next(iter(sliced.batch(3)))
      self.assertCountEqual(batch, ['a', 'b'])
      npt.assert_array_equal(batch['a'], [4, 6, 8])
      npt.assert_array_equal(batch['b'], [[4, 5], [6, 7], [8, 9]])


if __name__ == '__main__':
  absltest.main()
