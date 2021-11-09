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
"""Microbenchmarks for batch iteration speed of ClientDataset.

Benchmark results on a Xeon E5-1650 v3 on 2021/04/27:
ClientDataset   mode=train      preprocess=False         0.8456923319026828
TF Dataset      mode=train      preprocess=False         5.315499668009579
ClientDataset   mode=train      preprocess=True  0.9447170910425484
TF Dataset      mode=train      preprocess=True  5.322665546089411
ClientDataset   mode=eval       preprocess=False         0.015498528024181724
TF Dataset      mode=eval       preprocess=False         0.8721390531864017
ClientDataset   mode=eval       preprocess=True  0.021079374011605978
TF Dataset      mode=eval       preprocess=True  1.9080286680255085
"""

import timeit

from absl import app
from fedjax.core import client_datasets
from fedjax.core import util
import numpy as np

tf = util.import_tf()

# pylint: disable=cell-var-from-loop


def main(_):
  for mode in ['train', 'eval']:
    for preprocess in [False, True]:
      assert (bench_client_dataset(preprocess,
                                   mode) == bench_tf_dataset(preprocess, mode))
      print(
          f'ClientDataset\tmode={mode}\tpreprocess={preprocess}\t',
          timeit.timeit(
              lambda: bench_client_dataset(preprocess, mode), number=100))
      print(
          f'TF Dataset\tmode={mode}\tpreprocess={preprocess}\t',
          timeit.timeit(lambda: bench_tf_dataset(preprocess, mode), number=100))


FAKE_MNIST = {
    'pixels': np.random.uniform(size=(1000, 28, 28)),
    'label': np.random.randint(10, size=(1000,))
}


def f(x):
  return {**x, 'binary_label': x['label'] % 2}


def bench_client_dataset(preprocess, mode, batch_size=128, num_steps=100):
  """Benchmarks ClientDataset."""
  preprocessor = client_datasets.NoOpBatchPreprocessor
  if preprocess:
    preprocessor = preprocessor.append(f)
  dataset = client_datasets.ClientDataset(FAKE_MNIST, preprocessor)
  if mode == 'train':
    batches = dataset.shuffle_repeat_batch(
        batch_size=batch_size, num_steps=num_steps)
  else:
    batches = dataset.padded_batch(
        batch_size=batch_size, num_batch_size_buckets=4)
  n = 0
  for _ in batches:
    n += 1
  return n


def bench_tf_dataset(preprocess, mode, batch_size=128, num_steps=100):
  """Benchmarks TF Dataset."""
  shuffle_buffer = 1000  # size of FAKE_MNIST
  dataset = tf.data.Dataset.from_tensor_slices(FAKE_MNIST)
  if mode == 'train':
    dataset = dataset.shuffle(shuffle_buffer).repeat()
  dataset = dataset.batch(batch_size)
  if mode == 'train':
    dataset = dataset.take(num_steps)
  if preprocess:
    dataset = dataset.map(f)
  n = 0
  for _ in dataset.as_numpy_iterator():
    n += 1
  return n


if __name__ == '__main__':
  app.run(main)
