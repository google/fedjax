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
      npt.assert_equal(sliced, {'a': [0, 1, 2], 'b': [[0, 1], [2, 3], [4, 5]]})

    with self.subTest('Slicing like [3:6]'):
      sliced = client_datasets.slice_examples(examples, slice(3, 6))
      npt.assert_equal(sliced, {
          'a': [3, 4, 5],
          'b': [[6, 7], [8, 9], [10, 11]]
      })

    with self.subTest('Slicing like [3:6:2]'):
      sliced = client_datasets.slice_examples(examples, slice(3, 6, 2))
      npt.assert_equal(sliced, {'a': [3, 5], 'b': [[6, 7], [10, 11]]})

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
      npt.assert_equal(
          padded, {
              'a': [0, 1, 2, 3, 4, 0],
              'b': [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [0, 0]],
              'mask': [True, True, True, True, True, False]
          })

  def test_concat_examples(self):
    with self.subTest('empty'):
      self.assertDictEqual(client_datasets.concat_examples([]), {})

    with self.subTest('non-empty'):
      result = client_datasets.concat_examples([{
          'x': np.arange(5)
      }, {
          'x': np.arange(5, 10)
      }])
      npt.assert_equal(result, {'x': np.arange(10)})


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
      npt.assert_equal(
          result, {
              'pixels': fake_emnist['pixels'].reshape([-1, 28 * 28]),
              'label': fake_emnist['label'],
              'binary_label': fake_emnist['label'] % 2
          })

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
    self.assertIs(result['label'], fake_emnist['label'])
    npt.assert_equal(
        result, {
            'pixels': fake_emnist['pixels'].reshape([-1, 28 * 28]),
            'label': fake_emnist['label'],
            'binary_label': fake_emnist['label'] % 2,
            'sum_pixels': np.sum(fake_emnist['pixels'], axis=(1, 2))
        })


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
    view = d.batch(client_datasets.BatchHParams(batch_size=3))
    # `view` should be repeatedly iterable.
    for _ in range(2):
      batches = list(view)
      self.assertLen(batches, 2)
      npt.assert_equal(batches[0], {
          'a': [0, 2, 4],
          'b': [[0, 1], [2, 3], [4, 5]]
      })
      npt.assert_equal(batches[1], {'a': [6, 8], 'b': [[6, 7], [8, 9]]})

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
    view = d.batch(
        client_datasets.BatchHParams(
            batch_size=3, num_batch_size_buckets=1, mask_key='M'))
    # `view` should be repeatedly iterable.
    for _ in range(2):
      batches = list(view)
      self.assertLen(batches, 2)
      npt.assert_equal(batches[0], {
          'a': [0, 2, 4],
          'b': [[0, 1], [2, 3], [4, 5]]
      })
      npt.assert_equal(batches[1], {
          'a': [6, 8, 0],
          'b': [[6, 7], [8, 9], [0, 0]],
          'M': [True, True, False]
      })

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
      self.assertLen(
          list(
              d.shuffle_repeat_batch(
                  client_datasets.ShuffleRepeatBatchHParams(
                      batch_size=5, num_epochs=1))), 1)
      self.assertLen(
          list(
              d.shuffle_repeat_batch(
                  client_datasets.ShuffleRepeatBatchHParams(
                      batch_size=3, num_epochs=1))), 2)
      self.assertLen(
          list(
              d.shuffle_repeat_batch(
                  client_datasets.ShuffleRepeatBatchHParams(
                      batch_size=1, num_epochs=1))), 5)
      self.assertLen(
          list(
              d.shuffle_repeat_batch(
                  client_datasets.ShuffleRepeatBatchHParams(
                      batch_size=5, num_steps=4))), 4)
      self.assertLen(
          list(
              d.shuffle_repeat_batch(
                  client_datasets.ShuffleRepeatBatchHParams(
                      batch_size=3, num_steps=4))), 4)
      self.assertLen(
          list(
              d.shuffle_repeat_batch(
                  client_datasets.ShuffleRepeatBatchHParams(
                      batch_size=1, num_steps=4))), 4)
      self.assertLen(
          list(
              d.shuffle_repeat_batch(
                  client_datasets.ShuffleRepeatBatchHParams(
                      batch_size=5, num_epochs=1, num_steps=4))), 1)
      self.assertLen(
          list(
              d.shuffle_repeat_batch(
                  client_datasets.ShuffleRepeatBatchHParams(
                      batch_size=3, num_epochs=1, num_steps=4))), 2)
      self.assertLen(
          list(
              d.shuffle_repeat_batch(
                  client_datasets.ShuffleRepeatBatchHParams(
                      batch_size=1, num_epochs=1, num_steps=4))), 4)
      # 100 is as good as forever.
      self.assertLen(
          list(
              itertools.islice(
                  d.shuffle_repeat_batch(
                      client_datasets.ShuffleRepeatBatchHParams(batch_size=3)),
                  100)), 100)

    # Check proper shuffling.
    with self.subTest('shuffling'):
      view = d.shuffle_repeat_batch(
          client_datasets.ShuffleRepeatBatchHParams(
              batch_size=3, num_steps=4, seed=1))
      # `view` should be repeatedly iterable.
      for _ in range(2):
        batches = list(view)
        self.assertLen(batches, 4)
        npt.assert_equal(batches[0], {
            'a': [4, 2, 8],
            'b': [[4, 5], [2, 3], [8, 9]]
        })
        npt.assert_equal(batches[1], {
            'a': [0, 6, 4],
            'b': [[0, 1], [6, 7], [4, 5]]
        })
        npt.assert_equal(batches[2], {
            'a': [8, 6, 0],
            'b': [[8, 9], [6, 7], [0, 1]]
        })
        npt.assert_equal(batches[3], {
            'a': [2, 6, 0],
            'b': [[2, 3], [6, 7], [0, 1]]
        })

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
      batch = next(
          iter(sliced.batch(client_datasets.BatchHParams(batch_size=3))))
      npt.assert_equal(batch, {'a': [0, 2, 4], 'b': [[0, 1], [2, 3], [4, 5]]})

    with self.subTest('slice [-3:]'):
      sliced = d[-3:]
      batch = next(
          iter(sliced.batch(client_datasets.BatchHParams(batch_size=3))))
      npt.assert_equal(batch, {'a': [4, 6, 8], 'b': [[4, 5], [6, 7], [8, 9]]})


class BatchClientDatasetsTest(absltest.TestCase):

  def test_empty(self):
    batches = list(
        client_datasets.batch_client_datasets(
            [], client_datasets.BatchHParams(batch_size=128)))
    self.assertListEqual(batches, [])

  def test_single_no_buckets(self):
    batches = list(
        client_datasets.batch_client_datasets(
            [client_datasets.ClientDataset({'x': np.arange(6)})],
            client_datasets.BatchHParams(batch_size=5)))
    self.assertLen(batches, 2)
    npt.assert_equal(batches[0], {'x': np.arange(5)})
    npt.assert_equal(batches[1], {'x': np.arange(5, 6)})

  def test_single_has_buckets(self):
    batches = list(
        client_datasets.batch_client_datasets(
            [client_datasets.ClientDataset({'x': np.arange(8)})],
            client_datasets.BatchHParams(
                batch_size=6, num_batch_size_buckets=4)))
    self.assertLen(batches, 2)
    npt.assert_equal(batches[0], {'x': np.arange(6)})
    npt.assert_equal(batches[1], {'x': [6, 7, 0], 'mask': [True, True, False]})

  def test_multi(self):
    batches = list(
        client_datasets.batch_client_datasets([
            client_datasets.ClientDataset({'x': np.arange(10)}),
            client_datasets.ClientDataset({'x': np.arange(10, 11)}),
            client_datasets.ClientDataset({'x': np.arange(11, 15)}),
            client_datasets.ClientDataset({'x': np.arange(15, 17)})
        ], client_datasets.BatchHParams(batch_size=4)))
    self.assertLen(batches, 5)
    npt.assert_equal(batches[0], {'x': [0, 1, 2, 3]})
    npt.assert_equal(batches[1], {'x': [4, 5, 6, 7]})
    npt.assert_equal(batches[2], {'x': [8, 9, 10, 11]})
    npt.assert_equal(batches[3], {'x': [12, 13, 14, 15]})
    npt.assert_equal(batches[4], {'x': [16]})

  def test_preprocessor(self):
    batches = list(
        client_datasets.batch_client_datasets([
            client_datasets.ClientDataset({'x': np.arange(6)},
                                          client_datasets.Preprocessor(
                                              [lambda x: {
                                                  'x': x['x'] + 1
                                              }]))
        ], client_datasets.BatchHParams(batch_size=5)))
    self.assertLen(batches, 2)
    npt.assert_equal(batches[0], {'x': np.arange(5) + 1})
    npt.assert_equal(batches[1], {'x': np.arange(5, 6) + 1})

  def test_different_preprocessors(self):
    with self.assertRaisesRegex(
        ValueError,
        'client_datasets should have the identical Preprocessor object'):
      list(
          client_datasets.batch_client_datasets([
              client_datasets.ClientDataset({'x': np.arange(10)},
                                            client_datasets.Preprocessor()),
              client_datasets.ClientDataset({'x': np.arange(10, 11)},
                                            client_datasets.Preprocessor())
          ], client_datasets.BatchHParams(batch_size=4)))

  def test_different_features(self):
    with self.assertRaisesRegex(
        ValueError, 'client_datasets should have identical features'):
      list(
          client_datasets.batch_client_datasets([
              client_datasets.ClientDataset({'x': np.arange(10)}),
              client_datasets.ClientDataset({'y': np.arange(10, 11)})
          ], client_datasets.BatchHParams(batch_size=4)))


if __name__ == '__main__':
  absltest.main()
