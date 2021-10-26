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
"""Federated Shakespeare."""

import functools
from typing import Optional, Tuple

from fedjax.core import client_datasets
from fedjax.core import federated_data
from fedjax.core import sqlite_federated_data
from fedjax.datasets import downloads
import numpy as np

SPLITS = ('train', 'test')


def cite():
  """Returns BibTeX citation for the dataset."""
  return """@inproceedings{mcmahan2017communication,
  title={Communication-efficient learning of deep networks from
decentralized data},
  author={McMahan, Brendan and Moore, Eider and Ramage, Daniel and
Hampson, Seth and y Arcas, Blaise Aguera},
  booktitle={Artificial Intelligence and Statistics},
  pages={1273--1282},
  year={2017},
  organization={PMLR}
}"""


def load_split(split: str,
               mode: str = 'sqlite',
               cache_dir: Optional[str] = None) -> federated_data.FederatedData:
  """Loads a shakespeare split.

  Features:

  - snippets: [N] bytes array of snippet text.

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
    path = downloads.maybe_download(
        f'https://storage.googleapis.com/gresearch/fedjax/shakespeare/shakespeare_{split}.sqlite',
        cache_dir)
    return sqlite_federated_data.SQLiteFederatedData.new(path)
  else:
    raise ValueError(f'Unsupported mode={mode!r}')


def load_data(
    sequence_length: int = 80,
    mode: str = 'sqlite',
    cache_dir: Optional[str] = None
) -> Tuple[federated_data.FederatedData, federated_data.FederatedData]:
  """Loads preprocessed shakespeare splits.

  Preprocessing is done using :meth:`fedjax.FederatedData.preprocess_client`
  and :meth:`preprocess_client`.

  Features (M below is possibly different from N in load_split):

  - x: [M, sequence_length] int32 input labels, in the range of [0,
    shakespeare.VOCAB_SIZE)
  - y: [M, sequence_length] int32 output labels, in the range of [0,
    shakespeare.VOCAB_SIZE)

  Args:
    sequence_length: The fixed sequence length after preprocessing.
    mode: 'sqlite'.
    cache_dir: Directory to cache files in 'sqlite' mode.

  Returns:
    A (train, held_out, test) tuple of federated data.
  """
  train = load_split('train', mode, cache_dir)
  test = load_split('test', mode, cache_dir)
  preprocess = functools.partial(
      preprocess_client, sequence_length=sequence_length)
  return (train.preprocess_client(preprocess),
          test.preprocess_client(preprocess))


def _build_look_up_table(vocab: bytes,
                         num_reserved: int) -> Tuple[np.ndarray, int]:
  """Builds a look-up table from a byte to its integer label.

  Args:
    vocab: bytes object listing the byte values to include in the vocabulary.
      The byte vocab[i] is assigned label `num_reserved + i`. If the same byte
      occurs multiple times, the index of the last occurrence is used.
    num_reserved: Number of labels to reserve in the beginning of the integer
      label domain. Bytes in `vocab` will not be mapped to [0, num_reserved).

  Returns:
    (table, vocab_size) tuple. `table` is simply a [256] ndarray containing the
    integer label for each byte. Bytes not in `vocab` are mapped to `vocab_size
    - 1`.
  """
  oov = num_reserved + len(vocab)
  vocab_size = oov + 1
  table = np.full([256], oov, dtype=np.int32)
  for i, c in enumerate(vocab):
    table[c] = num_reserved + i
  return table, vocab_size


# Vocabulary re-used from the Federated Learning for Text Generation tutorial.
# https://www.tensorflow.org/federated/tutorials/federated_learning_for_text_generation
TABLE, VOCAB_SIZE = _build_look_up_table(
    b'dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r',
    num_reserved=3)
OOV = VOCAB_SIZE - 1
# Reserved labels.
PAD = 0
BOS = 1
EOS = 2


def preprocess_client(client_id: federated_data.ClientId,
                      examples: client_datasets.Examples,
                      sequence_length: int) -> client_datasets.Examples:
  """Turns snippets into sequences of integer labels.

  Features (M below is possibly different from N in load_split):

  - x: [M, sequence_length] int32 input labels, in the range of [0,
    shakespeare.VOCAB_SIZE)
  - y: [M, sequence_length] int32 output labels, in the range of [0,
    shakespeare.VOCAB_SIZE)

  All snippets in a client dataset are first joined into a single sequence (with
  BOS/EOS added), and then split into pairs of `sequence_length` chunks for
  language model training. For example, with sequence_length=3,
  `[b'ABCD', b'E']` becomes::

    Input sequences:  [[BOS, A, B], [C, D, EOS],   [BOS, E, PAD]]
    Output seqeunces: [[A, B, C],   [D, EOS, BOS], [E, EOS, PAD]]

  Note: This is not equivalent to the `TensorFlow Federated text generation tutorial <https://www.tensorflow.org/federated/tutorials/federated_learning_for_text_generation#load_and_preprocess_the_federated_shakespeare_data>`_
  (The processing logic there loses ~1/sequence_length portion of the tokens).

  Args:
    client_id: Not used.
    examples: Unprocessed examples (e.g. from `load_split()`).
    sequence_length: The fixed sequence length after preprocessing.

  Returns:
    Processed examples.
  """
  del client_id
  snippets = examples['snippets']
  # Join all snippets into a single label sequence.
  joined_length = sum(len(i) + 2 for i in snippets)
  joined = np.zeros([joined_length], dtype=np.int32)
  offset = 0
  for i in snippets:
    joined[offset] = BOS
    joined[offset + 1:offset + 1 + len(i)] = TABLE[list(i)]
    joined[offset + 1 + len(i)] = EOS
    offset += len(i) + 2
  # Split into input/output sequences of size `sequence_length`.
  padded_length = ((joined_length - 1 + sequence_length - 1) //
                   sequence_length * sequence_length)
  input_labels = np.full([padded_length], PAD, dtype=np.int32)
  input_labels[:joined_length - 1] = joined[:-1]
  output_labels = np.full([padded_length], PAD, dtype=np.int32)
  output_labels[:joined_length - 1] = joined[1:]
  return {
      'x': input_labels.reshape([-1, sequence_length]),
      'y': output_labels.reshape([-1, sequence_length])
  }
