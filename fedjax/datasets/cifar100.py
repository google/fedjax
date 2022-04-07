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
"""Federated cifar100."""

import os.path
from typing import List, Optional, Tuple

from fedjax.core import client_datasets
from fedjax.core import federated_data
from fedjax.core import sqlite_federated_data
from fedjax.core import util
from fedjax.datasets import downloads

import numpy as np

tf = util.import_tf()

SPLITS = ('train', 'test')
_TFF_SQLITE_COMPRESSED_HEXDIGEST = '23d3916c9caa33395737ee560cc7cb77bbd05fc7b73647ee7be3a7e764172939'
_TFF_SQLITE_COMPRESSED_NUM_BYTES = 153019920
_FEDJAX_SQLITE_HEXDIGEST = {
    'train': 'a4dc2f4ac4c9b6e7bd5234a1b568389384c00c903dcd34001c3cf50a4a81c713',
    'test': '74416c6ee8f41b1086f0e0e3c5289ac6df5a641100df48e8b583411a55de891f'
}
_FEDJAX_SQLITE_NUM_BYTES = {'train': 140521472, 'test': 28135424}


def cite():
  """Returns BibTeX citation for the dataset."""
  return """@TECHREPORT{Krizhevsky09learningmultiple,
    author = {Alex Krizhevsky},
    title = {Learning multiple layers of features from tiny images},
    institution = {},
    year = {2009}
}"""


def _parse_tf_examples(vs: List[bytes]) -> client_datasets.Examples:
  tf_examples = tf.io.parse_example(
      vs,
      features={
          'coarse_label': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
          'image': tf.io.FixedLenFeature(shape=(32, 32, 3), dtype=tf.int64),
          'label': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
      })
  tf_examples['image'] = tf.cast(tf_examples['image'], dtype=tf.uint8)
  return tf.nest.map_structure(lambda t: t.numpy(), tf_examples)


def load_split(split: str,
               mode: str = 'sqlite',
               cache_dir: Optional[str] = None) -> federated_data.FederatedData:
  """Loads a cifar100 split.

  Features:

  - image: [N, 32, 32, 3] uint8 pixels.
  - coarse_label: [N] int64 coarse labels in the range [0, 20).
  - label: [N] int64 labels in the range [0, 100).

  Args:
    split: Name of the split. One of SPLITS.
    mode: 'sqlite'.
    cache_dir: Directory to cache files in 'sqlite' mode.

  Returns:
    FederatedData.
  """
  if split not in SPLITS:
    raise ValueError(f'Invalid split={split!r}')
  if cache_dir is not None and mode != 'sqlite':
    raise ValueError('Caching locally is only supported in "sqlite" mode')
  if mode == 'sqlite':
    # Download and decompress LZMA compressed SQLite file from TFF.
    compressed_path = downloads.maybe_download(
        'https://storage.googleapis.com/tff-datasets-public/cifar100.sqlite.lzma',
        cache_dir)
    # Validate that the original TFF file has not been updated.
    downloads.validate_file(compressed_path, _TFF_SQLITE_COMPRESSED_NUM_BYTES,
                            _TFF_SQLITE_COMPRESSED_HEXDIGEST)
    decompressed_path = downloads.maybe_lzma_decompress(compressed_path)
    path = os.path.join(
        os.path.dirname(decompressed_path),
        f'federated_cifar100_{split}.sqlite')
    if os.path.exists(path):
      downloads.log(f'Reusing cached file {path!r}')
    else:
      # Convert TFF "1 example per record" to "all client examples per record".
      with sqlite_federated_data.SQLiteFederatedDataBuilder(path) as builder:
        client_ids_examples = map(
            lambda c: (c[0], c[1].all_examples()),
            sqlite_federated_data.TFFSQLiteClientsIterator(
                decompressed_path, _parse_tf_examples, split))
        builder.add_many(client_ids_examples)
      # Validate that the final produced SQLite is consistent.
      downloads.validate_file(path, _FEDJAX_SQLITE_NUM_BYTES[split],
                              _FEDJAX_SQLITE_HEXDIGEST[split])
    return sqlite_federated_data.SQLiteFederatedData.new(path)
  else:
    raise ValueError(f'Unsupported mode={mode!r}')


def load_data(
    mode: str = 'sqlite',
    cache_dir: Optional[str] = None
) -> Tuple[federated_data.FederatedData, federated_data.FederatedData]:
  """Loads partially preprocessed cifar100 splits.

  Features:

  - x: [N, 32, 32, 3] uint8 pixels.
  - y: [N] int32 labels in the range [0, 100).

  Additional preprocessing (e.g. centering and normalizing) depends on whether
  a split is used for training or eval. For example,::

    import functools
    from fedjax.datasets import cifar100
    # Load partially preprocessed splits.
    train, test = cifar100.load_data()
    # Preprocessing for training.
    train_for_train = train.preprocess_batch(
        functools.partial(preprocess_batch, is_train=True))
    # Preprocessing for eval.
    train_for_eval = train.preprocess_batch(
        functools.partial(preprocess_batch, is_train=False))
    test = test.preprocess_batch(
        functools.partial(preprocess_batch, is_train=False))

  Features after this preprocessing:

  - x: [N, 32, 32, 3] float32 preprocessed pixels.
  - y: [N] int32 labels in the range [0, 100).

  Alternatively, you can apply the same preprocessing as TensorFlow Federated
  following tff.simulation.baselines.cifar100.create_image_classification_task.
  For example,::

    from fedjax.datasets import cifar100
    train, test = cifar100.load_data()
    train = train.preprocess_batch(preprocess_batch_tff)
    test = test.preprocess_batch(preprocess_batch_tff)

  Features after this preprocessing:

  - x: [N, 24, 24, 3] float32 preprocessed pixels.
  - y: [N] int32 labels in the range [0, 100).

  Note: ``preprocess_batch`` and ``preprocess_batch_tff`` are just convenience
  wrappers around :meth:`preprocess_image` and :meth:`preprocess_image_tff`,
  respectively, for use with :meth:`fedjax.FederatedData.preprocess_batch`.

  Args:
    mode: 'sqlite'.
    cache_dir: Directory to cache files in 'sqlite' mode.

  Returns:
    A (train, test) tuple of federated data.
  """
  train = load_split('train', mode, cache_dir)
  test = load_split('test', mode, cache_dir)
  return (train.preprocess_client(preprocess_client),
          test.preprocess_client(preprocess_client))


def preprocess_client(
    client_id: federated_data.ClientId,
    examples: client_datasets.Examples) -> client_datasets.Examples:
  del client_id
  return {'x': examples['image'], 'y': examples['label'].astype(np.int32)}


# Mean and stddev computed assuming RGB values lie in [0,1].
CIFAR100_PIXELS_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR100_PIXELS_INVERSE_STDDEV = (
    1 / np.array([0.2023, 0.1994, 0.2010], dtype=np.float32))


def preprocess_image(image: np.ndarray, is_train: bool) -> np.ndarray:
  """Augments and preprocesses CIFAR-100 images by cropping, flipping, and normalizing.

  Preprocessing procedure and values taken from
  `pytorch-cifar
  <https://github.com/kuangliu/pytorch-cifar/blob/49b7aa97b0c12fe0d4054e670403a16b6b834ddd/main.py#L30-L40>`_.

  Args:
    image: [N, 32, 32, 3] uint8 pixels.
    is_train: Whether we are preprocessing for training or eval.

  Returns:
    Processed [N, 32, 32, 3] float32 pixels.
  """
  if is_train:
    # Pad 4 zero pixels on all sides and then randomly crop back to (32, 32, 3).
    num_paddings = 4
    image = np.pad(
        image, [(0, 0), (num_paddings, num_paddings),
                (num_paddings, num_paddings), (0, 0)],
        mode='constant')
    # Start offsets for cropping.
    i, j = np.random.randint(num_paddings * 2, size=[2])
    image = image[:, i:i + 32, j:j + 32, :]
    # Random horizontal flip.
    if np.random.randint(2):
      image = np.flip(image, axis=-2)
  # Center and normalize.
  image = ((image.astype(np.float32) / 255 - CIFAR100_PIXELS_MEAN) *
           CIFAR100_PIXELS_INVERSE_STDDEV)
  return image


def preprocess_batch(examples: client_datasets.Examples,
                     is_train: bool) -> client_datasets.Examples:
  return {'x': preprocess_image(examples['x'], is_train), 'y': examples['y']}


def preprocess_image_tff(image: np.ndarray, crop_height: int, crop_width: int,
                         distort: bool) -> np.ndarray:
  """Augments and preprocesses CIFAR-100 images following TensorFlow Federated."""
  if crop_height < 1 or crop_width < 1 or crop_height > 32 or crop_width > 32:
    raise ValueError('The crop_height and crop_width must be between 1 and 32.')

  if distort:
    # Distort via random crops and flips.
    # Start offsets for cropping should match tf.image.random_crop.
    crop_shape = np.array((crop_height, crop_width, 3)).astype(np.int32)
    shape = np.array(image.shape).astype(np.int32)[1:]
    limit = shape - crop_shape + 1
    offset = np.random.uniform(
        high=np.iinfo(limit.dtype).max, size=shape.shape).astype(
            limit.dtype) % limit
    begin_i, begin_j, _ = offset
    end_i, end_j, _ = offset + crop_shape
    image = image[:, begin_i:end_i, begin_j:end_j, :]
    # Random horizontal flip should match tf.image.random_flip_left_right.
    if np.random.randint(2):
      image = np.flip(image, axis=-2)
  else:
    # Center crop should match tf.image.resize_with_crop_or_pad.
    height_offset = (32 - crop_height) // 2
    width_offset = (32 - crop_width) // 2
    image = image[:, height_offset:height_offset + crop_height,
                  width_offset:width_offset + crop_width, :]
  image = image.astype(np.float32)
  # Normalize should match tf.image.per_image_standardization.
  num_pixels = np.prod(image.shape[-3:], dtype=np.float32)
  image_mean = np.mean(image, axis=(-1, -2, -3), keepdims=True)
  image_std = np.std(image, axis=(-1, -2, -3), keepdims=True)
  image_adjusted_std = np.maximum(image_std, np.sqrt(num_pixels))
  return (image - image_mean) / image_adjusted_std


def preprocess_batch_tff(examples: client_datasets.Examples,
                         crop_height: int = 24,
                         crop_width: int = 24,
                         distort: bool = False) -> client_datasets.Examples:
  """Preprocesses CIFAR-100 examples following TensorFlow Federated.

  Preprocessing procedure and values taken from
  `tff.simulation.baselines.cifar100.create_image_classification_task
  <https://www.tensorflow.org/federated/api_docs/python/tff/simulation/baselines/cifar100/create_image_classification_task>`_.

  Args:
    examples: Batch of examples containing
      - x: [N, 32, 32, 3] uint8 pixels.
      - y: [N] int32 labels in the range [0, 100).
    crop_height: Desired height for cropping images. Must be between 1 and 32.
      Defaults to the value used in TensorFlow Federated.
    crop_width: Desired width for cropping images. Must be between 1 and 32.
      Defaults to the value used in TensorFlow Federated.
    distort: Whether to distort the image via random crops and flips. Defaults
      to center cropping.

  Returns:
    Processed batch of examples containing
      - x: [N, ``crop_height``, ``crop_width``, 3] float32 pixels.
      - y: [N] int32 labels in the range [0, 100).
  """
  return {
      'x':
          preprocess_image_tff(examples['x'], crop_height, crop_width, distort),
      'y':
          examples['y'],
  }
