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
"""Federated stackoverflow."""

import itertools
from typing import Callable, List, Optional, Tuple

from fedjax.core import client_datasets
from fedjax.core import federated_data
from fedjax.core import sqlite_federated_data
from fedjax.core import util
from fedjax.datasets import downloads
import numpy as np

tf = util.import_tf()

SPLITS = ('train', 'held_out', 'test')


def cite():
  """Returns BibTeX citation for the dataset."""
  return """@misc{stackoverflow2019,
 title={TensorFlow Federated Stack Overflow dataset},
  author={The TensorFlow Federated Authors.},
  year={2019},
}"""


def load_split(split: str,
               mode: str = 'sqlite',
               cache_dir: Optional[str] = None) -> federated_data.FederatedData:
  """Loads a stackoverflow split.

  All bytes arrays are stored with `dtype=np.object`.

  Features:

  - creation_date: [N] bytes array. Textual timestamp, e.g.
    b'2018-02-28 19:06:18.34 UTC'.
  - title: [N] bytes array. The title of a post.
  - score: [N] int64 array. The score of a post.
  - tags: [N] bytes array. '|' separated list of tags, e.g. b'mysql|join'.
  - tokens: [N] bytes array. Space separated list of tokens.
  - type: [N] bytes array. Either b'question' or b'answer'.

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
        f'https://storage.googleapis.com/gresearch/fedjax/stackoverflow/stackoverflow_{split}.sqlite',
        cache_dir)
    return sqlite_federated_data.SQLiteFederatedData.new(path)
  else:
    raise ValueError(f'Unsupported mode={mode!r}')


def load_data(
    mode: str = 'sqlite',
    cache_dir: Optional[str] = None
) -> Tuple[federated_data.FederatedData, federated_data.FederatedData,
           federated_data.FederatedData]:
  """Loads partially preprocessed stackoverflow splits.

  Features:

  - domain_id: [N] int32 domain id derived from type (question = 0; answer = 1).
  - tokens: [N] bytes array. Space separated list of tokens.

  To convert `tokens` into padded/truncated integer labels, use a
  StackoverflowTokenizer. For example,::

    from fedjax.core.datasets import stackoverflow
    # Load partially preprocessed splits.
    train, held_out, test = stackoverflow.load_data()
    # Apply tokenizer during batching.
    tokenizer = stackoverflow.StackoverflowTokenizer()
    train_max_length, eval_max_length = 20, 30
    train_for_train = train.preprocess_batch(
        tokenizer.as_preprocess_batch(train_max_length))
    train_for_eval = train.preprocess_batch(
        tokenizer.as_preprocess_batch(eval_max_length))
    held_out = held_out.preprocess_batch(
        tokenizer.as_preprocess_batch(eval_max_length))
    test = test.preprocess_batch(
        tokenizer.as_preprocess_batch(eval_max_length))

  Features after tokenization:

  - domain_id: Same as before.
  - x: [N, max_length] int32 array of padded/truncated input labels.
  - y: [N, max_length] int32 array of padded/truncated output labels.

  Args:
    mode: 'sqlite'.
    cache_dir: Directory to cache files in 'sqlite' mode.

  Returns:
    A (train, held_out, test) tuple of federated data.
  """
  train = load_split('train', mode, cache_dir)
  held_out = load_split('held_out', mode, cache_dir)
  test = load_split('test', mode, cache_dir)
  return (train.preprocess_client(preprocess_client),
          held_out.preprocess_client(preprocess_client),
          test.preprocess_client(preprocess_client))


def preprocess_client(
    client_id: federated_data.ClientId,
    examples: client_datasets.Examples) -> client_datasets.Examples:
  """Attaches domain id and drops features other than `tokens`."""
  del client_id
  return {
      'domain_id': (examples['type'] == b'answer').astype(np.int32),
      'tokens': examples['tokens']
  }


def default_vocab(default_vocab_size) -> List[str]:
  """Loads the deafult stackoverflow vocabulary."""
  path = 'gs://gresearch/fedjax/stackoverflow/stackoverflow.word_count'
  vocab = []
  with tf.io.gfile.GFile(path) as f:
    for line in itertools.islice(f, default_vocab_size):
      word, _ = line.split()
      vocab.append(word)
  return vocab


# TODO(wuke): Remove dependency on TensorFlow.
class StackoverflowTokenizer:
  """Tokenizer for the `tokens` feature in stackoverflow.

  See :meth:`load_data` for examples.
  """
  PAD = 0
  BOS = 1
  EOS = 2

  def __init__(self,
               vocab: Optional[List[str]] = None,
               default_vocab_size: Optional[int] = 10000,
               num_oov_buckets: int = 1):
    """Initializes a tokenizer.

    Args:
      vocab: Optional vocabulary. If specified, `default_vocab_size` is ignored.
        If None, `default_vocab_size` is used to load the standard vocabulary.
        This vocabulary should NOT have special tokens PAD, EOS, BOS, and OOV.
        The special tokens are added and handled automatically by the tokenizer.
        The preprocessed examples will have vocabulary size `len(vocab) + 3 +
        num_oov_buckets`.
      default_vocab_size: Number of words in the default vocabulary. This is
        only used when `vocab` is not specified. The preprocessed examples will
        have vocabulary size `default_vocab_size + 3 + num_oov_buckets`
        with 3 special labels: 0 (PAD), 1 (BOS), 2 (EOS), and `num_oov_buckets`
        OOV labels starting at `default_vocab_size + 3`.
      num_oov_buckets: Number of out of vocabulary buckets.
    """
    if vocab is None:
      # Load default vocabulary.
      vocab = default_vocab(default_vocab_size)
    with tf.device('cpu'):
      self._table = tf.lookup.StaticVocabularyTable(
          tf.lookup.KeyValueTensorInitializer(
              vocab, tf.range(len(vocab), dtype=tf.int64)),
          num_oov_buckets=num_oov_buckets)

  def create_token_to_ids_fn(self, max_length: int):
    """Creates a Tf function that tokenizes tokens.

    Args:
      max_length: The length to pad x/y sequences to. Sequences longer than this
        are also truncated to this length.

    Returns:
      A function that uses tensorflow ops to tokenize tokens.
    """
    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def token_to_ids(tokens):
      """Tokenizes tokens with TensorFlow ops."""
      words = tf.strings.split(tokens, sep=' ')
      # Add 3 to reserve 0 (pad), 1 (bos), and 2 (eos). Hence offseting other
      # ids by 3.
      token_ids = self._table.lookup(words) + 3
      # Pad bos/eos.
      batch_bos = tf.zeros_like(
          tokens[..., tf.newaxis], dtype=token_ids.dtype) + self.BOS
      batch_eos = tf.zeros_like(
          tokens[..., tf.newaxis], dtype=token_ids.dtype) + self.EOS
      token_ids = tf.concat([batch_bos, token_ids, batch_eos], axis=-1)
      # Turn into x and y
      x = tf.cast(token_ids[..., :-1], tf.int32)
      y = tf.cast(token_ids[..., 1:], tf.int32)
      # Turn into dense tensors.
      shape = tokens.shape + [max_length]
      return (x.to_tensor(self.PAD,
                          shape=shape), y.to_tensor(self.PAD, shape=shape))
    return token_to_ids

  def as_preprocess_batch(
      self, max_length: int
  ) -> Callable[[client_datasets.Examples], client_datasets.Examples]:
    """Creates a preprocess_batch function.

    Args:
      max_length: The length to pad x/y sequences to. Sequences longer than this
        are also truncated to this length.

    Returns:
      A function that can be used with FederatedData.preprocess_batch().
    """
    token_to_ids = self.create_token_to_ids_fn(max_length)

    def preprocess_batch(
        examples: client_datasets.Examples) -> client_datasets.Examples:
      with tf.device('cpu'):
        x, y = token_to_ids(examples['tokens'])
      result = {'x': x.numpy(), 'y': y.numpy()}
      if 'domain_id' in examples:
        result['domain_id'] = examples['domain_id']
      return result

    return preprocess_batch
