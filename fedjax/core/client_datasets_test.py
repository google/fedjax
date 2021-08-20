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
"""Tests for fedjax.core.client_datasets."""

import itertools

from absl.testing import absltest
from fedjax.core import client_datasets
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

  def test_attach_mask(self):
    examples = {'a': np.arange(5), 'b': np.arange(10).reshape([5, 2])}
    mask = np.array([True, False, True, False, False])
    with self.subTest('mask key already used'):
      with self.assertRaises(ValueError):
        client_datasets.attach_mask(
            {
                **examples, client_datasets.EXAMPLE_MASK_KEY: None
            }, mask)
    with self.subTest('success'):
      npt.assert_equal(
          client_datasets.attach_mask(examples, mask), {
              'a': [0, 1, 2, 3, 4],
              'b': [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
              '__mask__': [True, False, True, False, False]
          })

  def test_pad_examples(self):
    examples = {'a': np.arange(5), 'b': np.arange(10).reshape([5, 2])}
    with self.subTest('maskkey already used'):
      with self.assertRaises(ValueError):
        client_datasets.pad_examples(
            {
                **examples, client_datasets.EXAMPLE_MASK_KEY: None
            }, size=8)
    with self.subTest('size too small'):
      with self.assertRaises(ValueError):
        client_datasets.pad_examples(examples, size=4)
    with self.subTest('no padding needed'):
      padded = client_datasets.pad_examples(examples, 5)
      npt.assert_equal(
          padded, {
              'a': [0, 1, 2, 3, 4],
              'b': [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
              '__mask__': [True, True, True, True, True]
          })
    with self.subTest('padding needed'):
      padded = client_datasets.pad_examples(examples, 6)
      npt.assert_equal(
          padded, {
              'a': [0, 1, 2, 3, 4, 0],
              'b': [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [0, 0]],
              '__mask__': [True, True, True, True, True, False]
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


class BatchPreprocessorTest(absltest.TestCase):

  def test_preprocessor(self):
    preprocessor = client_datasets.BatchPreprocessor([
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
    preprocessor = client_datasets.BatchPreprocessor([
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
        }, client_datasets.BatchPreprocessor([lambda x: {
            **x, 'a': 2 * x['a']
        }]))
    with self.subTest('keep remainder, kwargs'):
      view = d.batch(batch_size=3)
      # `view` should be repeatedly iterable.
      for _ in range(2):
        batches = list(view)
        self.assertLen(batches, 2)
        npt.assert_equal(batches[0], {
            'a': [0, 2, 4],
            'b': [[0, 1], [2, 3], [4, 5]]
        })
        npt.assert_equal(batches[1], {'a': [6, 8], 'b': [[6, 7], [8, 9]]})
    with self.subTest('drop remainder, hparams'):
      view = d.batch(
          client_datasets.BatchHParams(batch_size=3, drop_remainder=True))
      # `view` should be repeatedly iterable.
      for _ in range(2):
        batches = list(view)
        self.assertLen(batches, 1)
        npt.assert_equal(batches[0], {
            'a': [0, 2, 4],
            'b': [[0, 1], [2, 3], [4, 5]]
        })
    with self.subTest('no op drop remainder, hparams and kwargs'):
      view = d.batch(
          client_datasets.BatchHParams(batch_size=5), drop_remainder=True)
      # `view` should be repeatedly iterable.
      for _ in range(2):
        batches = list(view)
        self.assertLen(batches, 1)
        npt.assert_equal(batches[0], {
            'a': [0, 2, 4, 6, 8],
            'b': [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
        })

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

  def test_padded_batch(self):
    d = client_datasets.ClientDataset(
        {
            'a': np.arange(5),
            'b': np.arange(10).reshape([5, 2])
        }, client_datasets.BatchPreprocessor([lambda x: {
            **x, 'a': 2 * x['a']
        }]))
    with self.subTest('1 bucket, kwargs'):
      view = d.padded_batch(batch_size=3)
      # `view` should be repeatedly iterable.
      for _ in range(2):
        batches = list(view)
        self.assertLen(batches, 2)
        npt.assert_equal(
            batches[0], {
                'a': [0, 2, 4],
                'b': [[0, 1], [2, 3], [4, 5]],
                '__mask__': [True, True, True],
            })
        npt.assert_equal(
            batches[1], {
                'a': [6, 8, 0],
                'b': [[6, 7], [8, 9], [0, 0]],
                '__mask__': [True, True, False]
            })
    with self.subTest('2 buckets, kwargs override'):
      view = d.padded_batch(
          client_datasets.PaddedBatchHParams(batch_size=4),
          num_batch_size_buckets=2)
      # `view` should be repeatedly iterable.
      for _ in range(2):
        batches = list(view)
        self.assertLen(batches, 2)
        npt.assert_equal(
            batches[0], {
                'a': [0, 2, 4, 6],
                'b': [[0, 1], [2, 3], [4, 5], [6, 7]],
                '__mask__': [True, True, True, True],
            })
        npt.assert_equal(batches[1], {
            'a': [8, 0],
            'b': [[8, 9], [0, 0]],
            '__mask__': [True, False]
        })

  def test_shuffle_repeat_batch(self):
    d = client_datasets.ClientDataset(
        {
            'a': np.arange(5),
            'b': np.arange(10).reshape([5, 2])
        }, client_datasets.BatchPreprocessor([lambda x: {
            **x, 'a': 2 * x['a']
        }]))
    # Number of batches under different num_epochs/num_steps combinations.
    with self.subTest('repeating'):
      self.assertLen(list(d.shuffle_repeat_batch(batch_size=5)), 1)
      self.assertLen(list(d.shuffle_repeat_batch(batch_size=3)), 2)
      self.assertLen(list(d.shuffle_repeat_batch(batch_size=1)), 5)

      self.assertEmpty(
          list(d.shuffle_repeat_batch(batch_size=7, drop_remainder=True)))
      self.assertLen(
          list(d.shuffle_repeat_batch(batch_size=5, drop_remainder=True)), 1)
      self.assertLen(
          list(d.shuffle_repeat_batch(batch_size=3, drop_remainder=True)), 1)
      self.assertLen(
          list(d.shuffle_repeat_batch(batch_size=2, drop_remainder=True)), 2)
      self.assertLen(
          list(d.shuffle_repeat_batch(batch_size=1, drop_remainder=True)), 5)

      self.assertLen(
          list(
              d.shuffle_repeat_batch(
                  batch_size=5, num_epochs=None, num_steps=4)), 4)
      self.assertLen(
          list(
              d.shuffle_repeat_batch(
                  batch_size=3, num_epochs=None, num_steps=4)), 4)
      self.assertLen(
          list(
              d.shuffle_repeat_batch(
                  batch_size=1, num_epochs=None, num_steps=4)), 4)

      self.assertLen(
          list(
              d.shuffle_repeat_batch(
                  batch_size=5,
                  num_epochs=None,
                  num_steps=4,
                  drop_remainder=True)), 4)
      self.assertLen(
          list(
              d.shuffle_repeat_batch(
                  batch_size=3,
                  num_epochs=None,
                  num_steps=4,
                  drop_remainder=True)), 4)
      self.assertLen(
          list(
              d.shuffle_repeat_batch(
                  batch_size=1,
                  num_epochs=None,
                  num_steps=4,
                  drop_remainder=True)), 4)

      self.assertLen(list(d.shuffle_repeat_batch(batch_size=5, num_steps=4)), 1)
      self.assertLen(list(d.shuffle_repeat_batch(batch_size=3, num_steps=4)), 2)
      self.assertLen(list(d.shuffle_repeat_batch(batch_size=1, num_steps=4)), 4)

      self.assertLen(
          list(
              d.shuffle_repeat_batch(
                  batch_size=5, num_steps=4, drop_remainder=True)), 1)
      self.assertLen(
          list(
              d.shuffle_repeat_batch(
                  batch_size=3, num_steps=4, drop_remainder=True)), 1)
      self.assertLen(
          list(
              d.shuffle_repeat_batch(
                  batch_size=1, num_steps=4, drop_remainder=True)), 4)

      for drop_remainder in [False, True]:
        # 100 is as good as forever.
        self.assertLen(
            list(
                itertools.islice(
                    d.shuffle_repeat_batch(
                        batch_size=3,
                        num_epochs=None,
                        drop_remainder=drop_remainder), 100)), 100)

    # Check proper shuffling.
    with self.subTest('shuffling'):
      view = d.shuffle_repeat_batch(
          batch_size=3, num_epochs=None, num_steps=4, seed=1)
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

    with self.subTest('skip shuffling'):
      view = d.shuffle_repeat_batch(batch_size=3, skip_shuffle=True)
      batches = list(view)
      self.assertLen(batches, 2)
      # Original order should be maintained and loop back to beginning for fill.
      npt.assert_equal(batches[0], {
          'a': [0, 2, 4],
          'b': [[0, 1], [2, 3], [4, 5]]
      })
      npt.assert_equal(batches[1], {
          'a': [6, 8, 0],
          'b': [[6, 7], [8, 9], [0, 1]]
      })

  def test_slice(self):
    d = client_datasets.ClientDataset(
        {
            'a': np.arange(5),
            'b': np.arange(10).reshape([5, 2])
        }, client_datasets.BatchPreprocessor([lambda x: {
            **x, 'a': 2 * x['a']
        }]))

    with self.subTest('slice [:3]'):
      sliced = d[:3]
      batch = next(iter(sliced.batch(batch_size=3)))
      npt.assert_equal(batch, {'a': [0, 2, 4], 'b': [[0, 1], [2, 3], [4, 5]]})

    with self.subTest('slice [-3:]'):
      sliced = d[-3:]
      batch = next(iter(sliced.batch(batch_size=3)))
      npt.assert_equal(batch, {'a': [4, 6, 8], 'b': [[4, 5], [6, 7], [8, 9]]})

  def test_all_examples(self):
    raw_examples = {'a': np.arange(3), 'b': np.arange(6).reshape([3, 2])}
    with self.subTest('no preprocessing'):
      npt.assert_equal(
          client_datasets.ClientDataset(raw_examples).all_examples(),
          raw_examples)
    with self.subTest('with preprocessing'):
      npt.assert_equal(
          client_datasets.ClientDataset(
              raw_examples,
              client_datasets.BatchPreprocessor([lambda x: {
                  'c': x['a'] + 1
              }])).all_examples(), {'c': [1, 2, 3]})


class PaddedBatchClientDatasetsTest(absltest.TestCase):

  def test_empty(self):
    batches = list(
        client_datasets.padded_batch_client_datasets([], batch_size=128))
    self.assertListEqual(batches, [])

  def test_single_default_buckets(self):
    batches = list(
        client_datasets.padded_batch_client_datasets(
            [client_datasets.ClientDataset({'x': np.arange(6)})], batch_size=5))
    self.assertLen(batches, 2)
    npt.assert_equal(batches[0], {
        'x': np.arange(5),
        '__mask__': [True, True, True, True, True]
    })
    npt.assert_equal(batches[1], {
        'x': [5, 0, 0, 0, 0],
        '__mask__': [True, False, False, False, False]
    })

  def test_single_has_buckets(self):
    batches = list(
        client_datasets.padded_batch_client_datasets(
            [client_datasets.ClientDataset({'x': np.arange(8)})],
            batch_size=6,
            num_batch_size_buckets=4))
    self.assertLen(batches, 2)
    npt.assert_equal(batches[0], {
        'x': np.arange(6),
        '__mask__': [True, True, True, True, True, True]
    })
    npt.assert_equal(batches[1], {
        'x': [6, 7, 0],
        '__mask__': [True, True, False]
    })

  def test_multi(self):
    batches = list(
        client_datasets.padded_batch_client_datasets([
            client_datasets.ClientDataset({'x': np.arange(10)}),
            client_datasets.ClientDataset({'x': np.arange(10, 11)}),
            client_datasets.ClientDataset({'x': np.arange(11, 15)}),
            client_datasets.ClientDataset({'x': np.arange(15, 17)})
        ],
                                                     batch_size=4))
    self.assertLen(batches, 5)
    npt.assert_equal(batches[0], {
        'x': [0, 1, 2, 3],
        '__mask__': [True, True, True, True]
    })
    npt.assert_equal(batches[1], {
        'x': [4, 5, 6, 7],
        '__mask__': [True, True, True, True]
    })
    npt.assert_equal(batches[2], {
        'x': [8, 9, 10, 11],
        '__mask__': [True, True, True, True]
    })
    npt.assert_equal(batches[3], {
        'x': [12, 13, 14, 15],
        '__mask__': [True, True, True, True]
    })
    npt.assert_equal(batches[4], {
        'x': [16, 0, 0, 0],
        '__mask__': [True, False, False, False]
    })

  def test_preprocessor(self):
    batches = list(
        client_datasets.padded_batch_client_datasets([
            client_datasets.ClientDataset({'x': np.arange(6)},
                                          client_datasets.BatchPreprocessor(
                                              [lambda x: {
                                                  'x': x['x'] + 1
                                              }]))
        ],
                                                     batch_size=5))
    self.assertLen(batches, 2)
    npt.assert_equal(batches[0], {
        'x': np.arange(5) + 1,
        '__mask__': [True, True, True, True, True]
    })
    npt.assert_equal(batches[1], {
        'x': [6, 0, 0, 0, 0],
        '__mask__': [True, False, False, False, False]
    })

  def test_different_preprocessors(self):
    with self.assertRaisesRegex(
        ValueError,
        'client_datasets should have the identical Preprocessor object'):
      list(
          client_datasets.padded_batch_client_datasets([
              client_datasets.ClientDataset(
                  {'x': np.arange(10)}, client_datasets.BatchPreprocessor()),
              client_datasets.ClientDataset({'x': np.arange(10, 11)},
                                            client_datasets.BatchPreprocessor())
          ],
                                                       batch_size=4))

  def test_different_features(self):
    with self.assertRaisesRegex(
        ValueError, 'client_datasets should have identical features'):
      list(
          client_datasets.padded_batch_client_datasets([
              client_datasets.ClientDataset({'x': np.arange(10)}),
              client_datasets.ClientDataset({'y': np.arange(10, 11)})
          ],
                                                       batch_size=4))


class BufferedShuffleTest(absltest.TestCase):

  def test_buffered_shuffle(self):
    self.assertListEqual(
        list(
            client_datasets.buffered_shuffle(
                range(20), buffer_size=10, rng=np.random.RandomState(0))),
        [2, 1, 3, 7, 0, 13, 8, 12, 11, 17, 14, 15, 4, 9, 10, 6, 16, 18, 19, 5])


class BufferedShuffleBatchClientDatasetsTest(absltest.TestCase):

  def test_empty(self):
    batches = list(
        client_datasets.buffered_shuffle_batch_client_datasets(
            [], batch_size=5, buffer_size=10, rng=np.random.RandomState(0)))
    self.assertListEqual(batches, [])

  def test_single_buffer_1(self):
    batches = list(
        client_datasets.buffered_shuffle_batch_client_datasets(
            [client_datasets.ClientDataset({'x': np.arange(6)})],
            batch_size=5,
            buffer_size=1,
            rng=np.random.RandomState(0)))
    self.assertLen(batches, 2)
    # No shuffling.
    npt.assert_equal(batches[0], {'x': np.arange(5)})
    npt.assert_equal(batches[1], {'x': [5]})

  def test_single_buffer_4(self):
    batches = list(
        client_datasets.buffered_shuffle_batch_client_datasets(
            [client_datasets.ClientDataset({'x': np.arange(8)})],
            batch_size=6,
            buffer_size=4,
            rng=np.random.RandomState(0)))
    self.assertLen(batches, 2)
    npt.assert_equal(batches[0], {
        'x': [2, 4, 5, 6, 7, 3],
    })
    npt.assert_equal(batches[1], {
        'x': [1, 0],
    })

  def test_multi(self):
    batches = list(
        client_datasets.buffered_shuffle_batch_client_datasets(
            [
                client_datasets.ClientDataset({'x': np.arange(10)}),
                client_datasets.ClientDataset({'x': np.arange(10, 11)}),
                client_datasets.ClientDataset({'x': np.arange(11, 15)}),
                client_datasets.ClientDataset({'x': np.arange(15, 17)})
            ],
            batch_size=4,
            buffer_size=16,
            rng=np.random.RandomState(0)))
    self.assertLen(batches, 5)
    npt.assert_equal(batches[0], {
        'x': [1, 6, 16, 8],
    })
    npt.assert_equal(batches[1], {
        'x': [9, 13, 4, 2],
    })
    npt.assert_equal(batches[2], {
        'x': [14, 10, 7, 15],
    })
    npt.assert_equal(batches[3], {
        'x': [11, 3, 0, 5],
    })
    npt.assert_equal(batches[4], {
        'x': [12],
    })

  def test_preprocessor(self):
    batches = list(
        client_datasets.buffered_shuffle_batch_client_datasets(
            [
                client_datasets.ClientDataset({'x': np.arange(6)},
                                              client_datasets.BatchPreprocessor(
                                                  [lambda x: {
                                                      'x': x['x'] + 1
                                                  }]))
            ],
            batch_size=5,
            buffer_size=16,
            rng=np.random.RandomState(0)))
    self.assertLen(batches, 2)
    npt.assert_equal(batches[0], {
        'x': [6, 3, 2, 4, 1],
    })
    npt.assert_equal(batches[1], {
        'x': [5],
    })

  def test_different_preprocessors(self):
    with self.assertRaisesRegex(
        ValueError,
        'client_datasets should have the identical Preprocessor object'):
      list(
          client_datasets.buffered_shuffle_batch_client_datasets(
              [
                  client_datasets.ClientDataset(
                      {'x': np.arange(10, 20)},
                      client_datasets.BatchPreprocessor()),
                  client_datasets.ClientDataset(
                      {'x': np.arange(20, 30)},
                      client_datasets.BatchPreprocessor())
              ],
              batch_size=4,
              buffer_size=16,
              rng=np.random.RandomState(0)))

  def test_different_features(self):
    with self.assertRaisesRegex(
        ValueError, 'client_datasets should have identical features'):
      list(
          client_datasets.buffered_shuffle_batch_client_datasets(
              [
                  client_datasets.ClientDataset({'x': np.arange(10)}),
                  client_datasets.ClientDataset({'y': np.arange(10, 11)})
              ],
              batch_size=4,
              buffer_size=16,
              rng=np.random.RandomState(0)))


if __name__ == '__main__':
  absltest.main()
