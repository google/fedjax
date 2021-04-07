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
"""Preprocessing and batching operations over client datasets.

#   Column based representation

The examples in a client dataset can be viewed as a table, where the rows are
the individual examples, and the columns are the features (labels are viewed as
a feature in this context).

We use a column based representation when loading a dataset into memory.
-   Each column is a NumPy array `x` of rank at least 1, where `x[i, ...]` is
the value of this feature for the `i`-th example.
-   The complete set of examples is a dict-like object, from `str` feature
names, to the corresponding column values.

Traditionally, a row based representation is used for representing the entire
dataset, and a column based representation is used for a single batch. In the
context of federated learning, an individual client dataset is small enough to
easily fit into memory so the same representation is used for the entire dataset
and a batch.

#   Preprocessor

Preprocessing on a batch of examples can be easily done via a chain of
functions. A `Preprocessor` object holds the chain of functions, and applies the
transformation on a batch of examples.

#   ClientDataset: examples + preprocessor

A ClientDataset is simply some examples in the column based representation,
accompanied by a Preprocessor. Its `batch()` method produces batches of examples
in a sequential order, suitable for evaluation. Its `shuffle_repeat_batch()`
method adds shuffling and repeating, making it suitable for training.
"""

from typing import Callable, Iterable, Iterator, Mapping, Optional

import numpy as np

# The same column based representation for examples in the entire client
# dataset, or in a single batch.
Examples = Mapping[str, np.ndarray]


def assert_consistent_rows(examples: Examples):
  """Asserts `examples` have consistent row sizes (the number of examples)."""
  sizes = {k: v.shape[0] for k, v in examples.items()}
  if not sizes:
    raise ValueError('No features in examples')
  it = iter(sizes.items())
  name, size = next(it)
  for k, v in it:
    if v != size:
      raise ValueError(
          f'Feature {name} has {size} rows, but feature {k} has {v} rows')


def num_examples(examples: Examples, validate: bool = True) -> int:
  if validate:
    assert_consistent_rows(examples)
  return len(next(iter(examples.values())))


def slice_examples(examples: Examples, index: slice) -> Examples:
  return {k: v[index] for k, v in examples.items()}


def pad_examples(examples: Examples, size: int, mask_key: str) -> Examples:
  """Pad examples to a fixed number of rows with 0 values.

  Args:
    examples: Examples to pad.
    size: The desired number of rows (i.e. the leading dimension size).
    mask_key: Name of the new feature storing whether each row is a non-padding
      example.

  Returns:
    Padded examples.

  Raises:
    ValueError: invalid inputs.
  """
  if mask_key in examples:
    raise ValueError(
        f'mask_key {mask_key!r} is already present in examples {examples}')
  current_size = num_examples(examples)
  if current_size == size:
    return examples
  elif current_size > size:
    raise ValueError(f'Cannot pad {current_size} examples to size {size}')
  result = {mask_key: np.arange(size) < current_size}
  for k, v in examples.items():
    padded = np.zeros((size,) + v.shape[1:], v.dtype)
    padded[:current_size] = v
    result[k] = padded
  return result


class Preprocessor:
  """A chain of preprocessing functions.

  `Preprocessor` holds a chain of preprocessing functions, and applies them
  in order on batched examples. Each individual preprocessing function operates
  over multiple examples, instead of just 1 example. For example,

  ```
  preprocessor = Preprocessor([
    # Flattens `pixels`.
    lambda x: {**x, 'pixels': x['pixels'].reshape([-1, 28 * 28])},
    # Introduce `binary_label`.
    lambda x: {**x, 'binary_label': x['label'] % 2},
  ])
  fake_emnist = {
    'pixels': np.random.uniform(size=(10, 28, 28)),
    'label': np.random.randint(10, size=(10,))
  }
  preprocessor(fake_emnist)
  # Produces a dict of [10, 28*28] "pixels", [10,] "label" and "binary_label".
  ```

  Given a `Preprocessor`, a new `Preprocessor` can be created with an additional
  preprocessing function appended to the chain,

  ```
  # Continuing from the previous example.
  new_preprocessor = preprocessor.append(
    lambda x: {**x, 'sum_pixels': np.sum(x['pixels'], axis=1)})
  new_preprocessor(fake_emnist)
  # Produces a dict of [10, 28*28] "pixels", [10,] "sum_pixels", "label" and
  # "binary_label".
  ```

  `Preprocessor` can process either the entire dataset, or a batch, because of
  the identical representation. Certain preprocessing can be done either at the
  dataset level, or at the batch level.

  ### Examples of preprocessing possible at either the dataset level, or the
  batch level

  Such preprocessing is deterministic, and strictly per-example.

  -   Casting a feature from `int8` to `float32`.
  -   Adding a new feature derived from existing features.
  -   Remove a feature (although the better place to do so is at the dataset
  level).

  A simple rule for deciding where to carry out the preprocessing in this case
  is the following,
  -   Does this make batching cheaper (e.g. removing features)? If so, do it at
  the dataset level.
  -   Otherwise, do it at the batch level.

  Assuming preprocessing time is linear in the number of examples, preprocessing
  at the batch level has the benefit of evenly distributing host compute work,
  which may overlap better with asynchronous JAX compute work on GPU/TPU.

  ### Examples of preprocessing only possible at the batch level

  -   Data augmentation (e.g. random cropping).
  -   Padding at the batch size dimension.

  ### Examples of preprocessing only possible at the dataset level
  -   Capping the number of examples.
  -   Altering what it means to be an example: e.g. in certain language model
  setups, sentences are concatenated and then split into equal sized chunks.
  """

  def __init__(self, fns: Iterable[Callable[[Examples], Examples]] = ()):
    self._fns = tuple(fns)

  def __call__(self, examples: Examples) -> Examples:
    if not self._fns:
      return examples
    # Make a copy to guard against fns that modify their input.
    out = dict(examples)
    for f in self._fns:
      out = f(out)
    assert_consistent_rows(out)
    return out

  def append(self, fn: Callable[[Examples], Examples]) -> 'Preprocessor':
    """Creates a new Preprocessor with fn added to the end."""
    return Preprocessor(self._fns + (fn,))

  def __str__(self) -> str:
    return f'Preprocessor({self._fns})'

  def __repr__(self) -> str:
    return str(self)


# A common default preprocessor that does nothing.
NoOpPreprocessor = Preprocessor()


class ClientDataset:
  """In memory client dataset backed by numpy ndarrays.

  Custom preprocessing on batches can be added via a preprocessor.

  This is only intended for efficient access to small datasets that fit in
  memory.
  """

  def __init__(self,
               examples: Examples,
               preprocessor: Preprocessor = NoOpPreprocessor):
    assert_consistent_rows(examples)
    self.examples = examples
    self.preprocessor = preprocessor

  def __len__(self) -> int:
    """Returns the number of examples in this dataset."""
    return num_examples(self.examples, validate=False)

  def __getitem__(self, index: slice) -> 'ClientDataset':
    """Returns a new ClientDataset with sliced examples."""
    if not isinstance(index, slice):
      raise ValueError(f'Only slicing is supported, got index {index!r}')
    return ClientDataset(
        slice_examples(self.examples, index), self.preprocessor)

  def batch(self,
            batch_size: int,
            num_batch_size_buckets: Optional[int] = None,
            mask_key: str = 'mask') -> Iterable[Examples]:
    """Produces preprocessed batches in a fixed sequential order.

    When the number of examples in the dataset is not a multiple of
    `batch_size`, the final batch may be smaller than `batch_size`.
    This may lead to a large number of JIT recompilations. This can be
    circumvented by padding the final batch to a small number of fixed sizes by
    specifying `num_batch_size_buckets`. If the final batch is padded, a new
    bool feature named `mask_key` is added so that each non-padding example is
    marked with True.

    We repeatedly halve the batch size up to `num_batch_size_buckets-1` times,
    until we find the smallest one that is also >= the size of the final batch.
    Therefore if `batch_size < 2^num_batch_size_buckets`, fewer bucket sizes
    will be actually used.

    Args:
      batch_size: Desired batch size.
      num_batch_size_buckets: Optional number of batch size buckets for the
        final batch.
      mask_key: The name of the new mask feature.

    Returns:
      An iterable object that can be iterated over multiple times.
    """
    return BatchView(self, batch_size, num_batch_size_buckets, mask_key)

  def shuffle_repeat_batch(self,
                           batch_size: int,
                           num_epochs: Optional[int] = None,
                           num_steps: Optional[int] = None,
                           seed: Optional[int] = None) -> Iterable[Examples]:
    """Produces preprocessed batches in a shuffled and repeated order.

    Shuffling is done without replacement, therefore for a dataset of N
    examples, the first `ceil(N/batch_size)` batches are guarranteed to cover
    the entire dataset.

    The number of batches produced from the iteration can be controlled by the
    `(num_epochs, num_steps)` iteration:
    -   If both are None, the shuffle-repeat process continues forever.
    -   If only `num_epochs` is set, as few batches as needed to go over the
    dataset this many passes are produced.
    -   If only `num_steps` is set, exactly this many batches are produced.
    -   If both `num_epochs` and `num_steps` are set, the fewer number of
    batches between the two conditions are produced.

    If reproducible iteration order is desired, a fixed `seed` can be used. When
    `seed` is None, repeated iteration over the same object may produce batches
    in a different order.

    Unlike `batch()`, batches from `shuffle_repeat_batch()` always contain
    exactly `batch_size` examples.

    Args:
      batch_size: The desired batch size.
      num_epochs: Optional number of passes to iterate over the dataset.
      num_steps: Optional number of batches to produce.
      seed: Optional random number generator seed.

    Returns:
      An iterable object that can be iterated over multiple times.
    """
    return ShuffleRepeatBatchView(
        self,
        batch_size=batch_size,
        num_epochs=num_epochs,
        num_steps=num_steps,
        seed=seed)


class BatchView:
  """View of ordered batches of a ClientDataset.

  See ClientDataset.batch() for the expected behavior.
  """

  def __init__(self, client_dataset: ClientDataset, batch_size: int,
               num_batch_size_buckets: Optional[int], mask_key: str):
    self._client_dataset = client_dataset
    self._data_size = len(client_dataset)
    self._batch_size = batch_size
    if num_batch_size_buckets is None:
      self._final_batch_size = None
    else:
      self._final_batch_size = _pick_final_batch_size(self._data_size,
                                                      self._batch_size,
                                                      num_batch_size_buckets)
    self._mask_key = mask_key

  def __iter__(self) -> Iterator[Examples]:
    for start in range(0, self._data_size, self._batch_size):
      stop = start + self._batch_size
      sliced = slice_examples(self._client_dataset.examples, slice(start, stop))
      processed = self._client_dataset.preprocessor(sliced)
      if stop <= self._data_size or self._final_batch_size is None:
        yield processed
      else:
        yield pad_examples(processed, self._final_batch_size, self._mask_key)


def _pick_final_batch_size(data_size: int, batch_size: int,
                           num_batch_size_buckets: int) -> int:
  """Picks the final batch size for a given dataset size."""
  # Determine the batch size for the final batch.
  final_batch_size = data_size % batch_size
  if final_batch_size == 0:
    # No padding necessary.
    return batch_size
  # final_batch_size in [1, batch_size)
  high, low, n = batch_size, batch_size // 2, 1
  # Find low < final_batch_size <= high
  while low >= final_batch_size and n < num_batch_size_buckets:
    high, low, n = low, low // 2, n + 1
  return high


class ShuffleRepeatBatchView:
  """View of shuffled and repeated batches of a ClientDataset.

  See ClientDataset.shuffle_repeat_batch() for the expected behavior.
  """

  def __init__(self, client_dataset: ClientDataset, batch_size: int,
               num_epochs: Optional[int], num_steps: Optional[int],
               seed: Optional[int]):
    self._client_dataset = client_dataset
    self._data_size = len(client_dataset)
    self._batch_size = batch_size
    if num_epochs is not None:
      self._num_steps = ((self._data_size * num_epochs + batch_size - 1) //
                         batch_size)
      if num_steps is not None:
        self._num_steps = min(num_steps, self._num_steps)
    elif num_steps is not None:
      self._num_steps = num_steps
    else:
      self._num_steps = None
    self._seed = seed

  def __iter__(self) -> Iterator[Examples]:
    buf = np.arange(self._data_size, dtype=np.int32)
    buf_size = buf.shape[0]
    # Start of unused portion of buf. We start with no unused values because we
    # haven't shuffled yet.
    i = buf_size
    num_steps = 0
    desired_num_steps = self._num_steps
    rng = np.random.RandomState(self._seed)
    while desired_num_steps is None or num_steps < desired_num_steps:
      # Find example indices of next batch.
      indices = np.zeros((self._batch_size,), dtype=np.int32)
      desired_size = indices.size
      filled = 0
      while filled < desired_size:
        available = buf_size - i
        if available == 0:
          rng.shuffle(buf)
          i = 0
          available = buf_size
        used = min(available, desired_size - filled)
        indices[filled:filled + used] = buf[i:i + used]
        i += used
        filled += used
      # Produce next batch.
      sliced = {k: v[indices] for k, v in self._client_dataset.examples.items()}
      yield self._client_dataset.preprocessor(sliced)
      num_steps += 1
