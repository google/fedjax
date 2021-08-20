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

**Column based representation**

The examples in a client dataset can be viewed as a table, where the rows are
the individual examples, and the columns are the features (labels are viewed as
a feature in this context).

We use a column based representation when loading a dataset into memory.

- Each column is a NumPy array ``x`` of rank at least 1, where ``x[i, ...]`` is
  the value of this feature for the ``i``-th example.
- The complete set of examples is a dict-like object, from ``str`` feature
  names, to the corresponding column values.

Traditionally, a row based representation is used for representing the entire
dataset, and a column based representation is used for a single batch. In the
context of federated learning, an individual client dataset is small enough to
easily fit into memory so the same representation is used for the entire dataset
and a batch.

**Preprocessor**

Preprocessing on a batch of examples can be easily done via a chain of
functions. A ``Preprocessor`` object holds the chain of functions, and applies
the transformation on a batch of examples.

**ClientDataset: examples + preprocessor**

A :class:`~fedjax.ClientDataset` is simply some examples in the column based
representation, accompanied by a Preprocessor.
Its :meth:`~fedjax.ClientDataset.batch` method produces batches of examples in a
sequential order, suitable for evaluation.
Its :meth:`~fedjax.ClientDataset.shuffle_repeat_batch` method adds shuffling and
repeating, making it suitable for training.
"""

import collections
import itertools
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional

from fedjax.core import dataclasses
import numpy as np

# The same column based representation for examples in the entire client
# dataset, or in a single batch.
Examples = Mapping[str, np.ndarray]


class BatchPreprocessor:
  """A chain of preprocessing functions on batched examples.

  BatchPreprocessor holds a chain of preprocessing functions, and applies them
  in order on batched examples. Each individual preprocessing function operates
  over multiple examples, instead of just 1 example. For example, ::

    preprocessor = BatchPreprocessor([
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

  Given a BatchPreprocessor, a new BatchPreprocessor can be created with an
  additional preprocessing function appended to the chain, ::

    # Continuing from the previous example.
    new_preprocessor = preprocessor.append(
      lambda x: {**x, 'sum_pixels': np.sum(x['pixels'], axis=1)})
    new_preprocessor(fake_emnist)
    # Produces a dict of [10, 28*28] "pixels", [10,] "sum_pixels", "label" and
    # "binary_label".

  The main difference of this preprocessor and
  :class:`fedjax.ClientPreprocessor` is that :class:`fedjax.ClientPreprocessor`
  also takes ``client_id`` as input. Because of the identical representation
  between batched examples and all examples in a client dataset, certain
  preprocessing can be done with either BatchPreprocessor or ClientPreprocessor.

  **Examples of preprocessing possible at either the client dataset level, or
  the batch level**

  Such preprocessing is deterministic, and strictly per-example.

  - Casting a feature from `int8` to `float32`.
  - Adding a new feature derived from existing features.
  - Remove a feature (although the better place to do so is at the dataset
    level).

  A simple rule for deciding where to carry out the preprocessing in this case
  is the following,

  - Does this make batching cheaper (e.g. removing features)? If so, do it at
    the dataset level.
  - Otherwise, do it at the batch level.

  Assuming preprocessing time is linear in the number of examples, preprocessing
  at the batch level has the benefit of evenly distributing host compute work,
  which may overlap better with asynchronous JAX compute work on GPU/TPU.

  **Examples of preprocessing only possible at the batch level**

  - Data augmentation (e.g. random cropping).
  - Padding at the batch size dimension.

  **Examples of preprocessing only possible at the dataset level**

  - Those that require knowing the client id.
  - Capping the number of examples.
  - Altering what it means to be an example: e.g. in certain language model
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

  def append(self, fn: Callable[[Examples], Examples]) -> 'BatchPreprocessor':
    """Creates a new BatchPreprocessor with fn added to the end."""
    return BatchPreprocessor(self._fns + (fn,))

  def __str__(self) -> str:
    return f'BatchPreprocessor({self._fns})'

  def __repr__(self) -> str:
    return str(self)


# A common default preprocessor that does nothing.
NoOpBatchPreprocessor = BatchPreprocessor()


# A special feature name for padded batches.
EXAMPLE_MASK_KEY = '__mask__'


# We group hyperparams to batching functions so that,
#
# 1.  They can be easily passed around as a single object, instead of a list of
#     arguments.
# 2.  The list of argument is only defined, documented, and assigned default
#     values once (in the hyperparams class).
#
# This setup is more convenient for library code where the caller specifies such
# a hyperparams object, but less so for users directly calling the actual
# batching function (e.g. compare `padded_batch(batch_size=3)` vs
# `padded_batch(PaddedBatchHParams(batch_size=3))`. Therefore, all our batching
# functions support two ways of invocation:
# -   They can be passed in a hyperparams object.
# -   They can also be passed in a list of keyword arguments, which will be used
#     to construct the hyperparams object.
#
# See ClientDataset.padded_batch() for an example.


@dataclasses.dataclass
class PaddedBatchHParams:
  """See :meth:`ClientDataset.padded_batch`.

  Attributes:
    batch_size: Desired batch size.
    num_batch_size_buckets: Number of batch size buckets for the final batch.
  """
  batch_size: int
  num_batch_size_buckets: int = 1


@dataclasses.dataclass
class ShuffleRepeatBatchHParams:
  """See :meth`:ClientDataset.shuffle_repeat_batch`.

  Attributes:
    batch_size: Desired batch size.
    num_epochs: Optional number of passes to iterate over the dataset.
    num_steps: Optional number of batches to produce.
    drop_remainder: Whether to drop a trailing batch smaller than ``batch_size``.
    seed: Optional random number generator seed.
    skip_shuffle: Whether to skip the shuffle step.
  """
  batch_size: int
  num_epochs: Optional[int] = 1
  num_steps: Optional[int] = None
  drop_remainder: bool = False
  seed: Optional[int] = None
  skip_shuffle: bool = False


@dataclasses.dataclass
class BatchHParams:
  """See :meth:`ClientDataset.batch`.

  Attributes:
    batch_size: Desired batch size.
    drop_remainder: Whether to drop the final batch when it contains fewer than
      ``batch_size`` examples.
  """
  batch_size: int
  drop_remainder: bool = False


class ClientDataset:
  """In memory client dataset backed by numpy ndarrays.

  Custom preprocessing on batches can be added via a preprocessor. A
  ClientDataset is stored as the unpreprocessed :attr:`raw_examples`, along with
  its preprocessor.

  - To access batches, use one of the batching functions (e.g.
    :meth:`shuffle_repeat_batch` for training, :meth:`padded_batch`
    for evaluation).
  - To access a small number of preprocessed examples (e.g. for exploration),
    use slicing + :meth:`all_examples`.

  This is only intended for efficient access to small datasets that fit in
  memory.
  """

  def __init__(self,
               raw_examples: Examples,
               preprocessor: BatchPreprocessor = NoOpBatchPreprocessor):
    assert_consistent_rows(raw_examples)
    self.raw_examples = raw_examples
    self.preprocessor = preprocessor

  def __len__(self) -> int:
    """Returns the number of raw examples in this dataset."""
    return num_examples(self.raw_examples, validate=False)

  def __getitem__(self, index: slice) -> 'ClientDataset':
    """Returns a new ClientDataset with sliced raw examples."""
    if not isinstance(index, slice):
      raise ValueError(f'Only slicing is supported, got index {index!r}')
    return ClientDataset(
        slice_examples(self.raw_examples, index), self.preprocessor)

  def all_examples(self) -> Examples:
    """Returns the result of feeding all raw examples through the preprocessor.

    This is mostly intended for interactive exploration of a small subset of a
    client dataset. For example, to see the first 4 examples in a client
    dataset, ::

      dataset = ClientDataset(my_raw_examples, my_preprocessor)
      dataset[:4].all_examples()

    Returns:
      Preprocessed examples from all the raw examples in this client dataset.
    """
    return self.preprocessor(self.raw_examples)

  def padded_batch(self,
                   hparams: Optional[PaddedBatchHParams] = None,
                   **kwargs) -> Iterable[Examples]:
    """Produces preprocessed padded batches in a fixed sequential order.

    This function can be invoked in 2 ways:

    1.  Using a hyperparams object. This is the recommended way in library code.
        Example::

          def a_library_function(client_dataset, hparams):
            for batch in client_dataset.padded_batch(hparams):
              ...

    2.  Using keyword arguments. The keyword arguments are used to construct
        a new hyperparams object, or override an existing one. For example, ::

          client_dataset.padded_batch(batch_size=2)
          # Overrides the default num_batch_size_buckets value.
          client_dataset.padded_batch(hparams, num_batch_size_buckets=2)

    When the number of examples in the dataset is not a multiple of
    ``batch_size``, the final batch may be smaller than ``batch_size``.
    This may lead to a large number of JIT recompilations. This can be
    circumvented by padding the final batch to a small number of fixed sizes
    controlled by ``num_batch_size_buckets``.

    All batches contain an extra bool feature keyed by ``EXAMPLE_MASK_KEY``.
    ``batch[EXAMPLE_MASK_KEY][i]`` tells us whether the ``i``-th example in this
    batch is an actual example (``batch[EXAMPLE_MASK_KEY][i] == True``), or a
    padding example (``batch[EXAMPLE_MASK_KEY][i] == False``).

    We repeatedly halve the batch size up to ``num_batch_size_buckets-1`` times,
    until we find the smallest one that is also >= the size of the final batch.
    Therefore if ``batch_size < 2^num_batch_size_buckets``, fewer bucket sizes
    will be actually used.

    Args:
      hparams: Batching hyperparameters.
      **kwargs: Keyword arguments for constructing/overriding hparams.

    Returns:
      An iterable object that can be iterated over multiple times.
    """
    if hparams is None:
      hparams = PaddedBatchHParams(**kwargs)
    elif kwargs:
      hparams = hparams.replace(**kwargs)
    return PaddedBatchView(self, hparams)

  def shuffle_repeat_batch(self,
                           hparams: Optional[ShuffleRepeatBatchHParams] = None,
                           **kwargs) -> Iterable[Examples]:
    """Produces preprocessed batches in a shuffled and repeated order.

    This function can be invoked in 2 ways:

    1.  Using a hyperparams object. This is the recommended way in library code.
        Example::

          def a_library_function(client_dataset, hparams):
            for batch in client_dataset.shuffle_repeat_batch(hparams):
              ...

    2.  Using keyword arguments. The keyword arguments are used to construct
        a new hyperparams object, or override an existing one. For example, ::

          client_dataset.shuffle_repeat_batch(batch_size=2)
          # Overrides the default num_epochs value.
          client_dataset.shuffle_repeat_batch(hparams, num_epochs=2)

    Shuffling is done without replacement, therefore for a dataset of N
    examples, the first ``ceil(N/batch_size)`` batches are guarranteed to cover
    the entire dataset.

    By default the iteration stops after the first epoch. The number of batches
    produced from the iteration can be controlled by the
    ``(num_epochs, num_steps, drop_remainder)`` combination:

    - If both ``num_epochs`` and ``num_steps`` are None, the shuffle-repeat
      process continues forever.
    - If ``num_epochs`` is set and ``num_steps`` is None, as few batches as needed
      to go over the dataset this many passes are produced. Further,

        - If ``drop_remainder`` is False (the default), the final batch is
          filled with additionally sampled examples to contain ``batch_size``
          examples.
        - If ``drop_remainder`` is True, the final batch is dropped if it
          contains fewer than ``batch_size`` examples. This may result in
          examples being skipped when ``num_epochs=1``.
    - If ``num_steps`` is set and ``num_steps`` is None, exactly this many batches
      are produced. ``drop_remainder`` has no effect in this case.
    - If both ``num_epochs`` and ``num_steps`` are set, the fewer number of
      batches between the two conditions are produced.

    If reproducible iteration order is desired, a fixed ``seed`` can be used. When
    ``seed`` is None, repeated iteration over the same object may produce batches
    in a different order.

    Unlike ``batch()`` or ``padded_batch()``, batches from ``shuffle_repeat_batch()``
    always contain exactly ``batch_size`` examples. Also unlike TensorFlow, that
    holds even when ``drop_remainder=False``.

    Args:
      hparams: Batching hyperparamters.
      **kwargs: Keyword arguments for constructing/overriding hparams.

    Returns:
      An iterable object that can be iterated over multiple times.
    """
    if hparams is None:
      hparams = ShuffleRepeatBatchHParams(**kwargs)
    elif kwargs:
      hparams = hparams.replace(**kwargs)
    return ShuffleRepeatBatchView(self, hparams)

  def batch(self,
            hparams: Optional[BatchHParams] = None,
            **kwargs) -> Iterable[Examples]:
    """Produces preprocessed batches in a fixed sequential order.

    The final batch may contain fewer than ``batch_size`` examples. If used
    directly, that may result in a large number of JIT recompilations. Therefore
    we recommended using ``padded_batch`` instead in most scenarios.

    This function can be invoked in 2 ways:

    1.  Using a hyperparams object. This is the recommended way in library code
        if you have to use batch (prefer padded_batch() if possible). Example::

          def a_library_function(client_dataset, hparams):
            for batch in client_dataset.batch(hparams):
              ...

    2.  Using keyword arguments. The keyword arguments are used to construct
        a new hyperparams object, or override an existing one. For example, ::

          client_dataset.batch(batch_size=2)
          # Overrides the default drop_remainder value.
          client_dataset.batch(hparams, drop_remainder=True)

    Args:
      hparams: Batching hyperparameters.
      **kwargs: Keyword arguments for constructing/overriding hparams.

    Returns:
      An iterable object that can be iterated over multiple times.
    """
    if hparams is None:
      hparams = BatchHParams(**kwargs)
    elif kwargs:
      hparams = hparams.replace(**kwargs)
    return BatchView(self, hparams)


class PaddedBatchView:
  """View of ordered padded batches of a ClientDataset.

  See ClientDataset.padded_batch() for the expected behavior.
  """

  def __init__(self, client_dataset: ClientDataset,
               hparams: PaddedBatchHParams):
    self._client_dataset = client_dataset
    self._data_size = len(client_dataset)
    self._batch_size = hparams.batch_size
    self._final_batch_size = _pick_final_batch_size(
        self._data_size, self._batch_size, hparams.num_batch_size_buckets)

  def __iter__(self) -> Iterator[Examples]:
    # Mask for full batches.
    full_mask = np.ones([self._batch_size], dtype=np.bool_)
    for start in range(0, self._data_size, self._batch_size):
      stop = start + self._batch_size
      sliced = slice_examples(self._client_dataset.raw_examples,
                              slice(start, stop))
      processed = self._client_dataset.preprocessor(sliced)
      if stop <= self._data_size:
        yield {**processed, EXAMPLE_MASK_KEY: full_mask}
      else:
        yield pad_examples(processed, self._final_batch_size)


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

  def __init__(self, client_dataset: ClientDataset,
               hparams: ShuffleRepeatBatchHParams):
    self._client_dataset = client_dataset
    self._data_size = len(client_dataset)
    self._batch_size = hparams.batch_size
    if hparams.num_epochs is not None:
      if hparams.drop_remainder:
        self._num_steps = (
            self._data_size * hparams.num_epochs // hparams.batch_size)
      else:
        self._num_steps = (
            (self._data_size * hparams.num_epochs + hparams.batch_size - 1) //
            hparams.batch_size)
      if hparams.num_steps is not None:
        self._num_steps = min(hparams.num_steps, self._num_steps)
    elif hparams.num_steps is not None:
      self._num_steps = hparams.num_steps
    else:
      self._num_steps = None
    self._seed = hparams.seed
    self._skip_shuffle = hparams.skip_shuffle

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
          if not self._skip_shuffle:
            rng.shuffle(buf)
          i = 0
          available = buf_size
        used = min(available, desired_size - filled)
        indices[filled:filled + used] = buf[i:i + used]
        i += used
        filled += used
      # Produce next batch.
      sliced = {
          k: v[indices] for k, v in self._client_dataset.raw_examples.items()
      }
      yield self._client_dataset.preprocessor(sliced)
      num_steps += 1


class BatchView:
  """View of ordered batches of a ClientDataset.

  See ClientDataset.batch() for the expected behavior.
  """

  def __init__(self, client_dataset: ClientDataset, hparams: BatchHParams):
    self._client_dataset = client_dataset
    self._batch_size = hparams.batch_size
    self._drop_remainder = hparams.drop_remainder
    self._data_size = len(client_dataset)

  def __iter__(self) -> Iterator[Examples]:
    for start in range(0, self._data_size, self._batch_size):
      stop = start + self._batch_size
      sliced = slice_examples(self._client_dataset.raw_examples,
                              slice(start, stop))
      processed = self._client_dataset.preprocessor(sliced)
      if not self._drop_remainder or stop <= self._data_size:
        yield processed


def padded_batch_client_datasets(datasets: Iterable[ClientDataset],
                                 hparams: Optional[PaddedBatchHParams] = None,
                                 **kwargs) -> Iterator[Examples]:
  """Batches examples from multiple client datasets.

  This is useful when we want to evaluate on the combined dataset consisting of
  multiple client datasets. Unlike batching each client dataset individually, we
  can reduce the number of batches smaller than ``batch_size``.

  This function can be invoked in 2 ways:

  1.  Using a hyperparams object. This is the recommended way in library code.
      Example::

        def a_library_function(datasets, hparams):
          for batch in padded_batch_client_datasets(datasets, hparams):
            ...

  2.  Using keyword arguments. The keyword arguments are used to construct
      a new hyperparams object, or override an existing one. For example, ::

        padded_batch_client_datasets(datasets, hparams)
        # Overrides the default num_batch_size_buckets value.
        padded_batch_client_datasets(datasets, hparams, num_batch_size_buckets=2)

  Args:
    datasets: ClientDatasets to be batched. All ClientDatasets must have the
      same Preprocessor object attached.
    hparams: Batching hyperparams like those in :meth:`ClientDataset.padded_batch`.
    **kwargs: Keyword arguments for constructing/overriding hparams.

  Yields:
    Batches of examples. The final batch might be padded. All batches contain
    a bool feature keyed by `EXAMPLE_MASK_KEY`.

  Raises:
    ValueError: If any 2 client datasets have different Preprocessors.
    ValueError: If any 2 client datasets have different features.
  """
  if hparams is None:
    hparams = PaddedBatchHParams(**kwargs)
  elif kwargs:
    hparams = hparams.replace(**kwargs)
  preprocessor = None
  features = None
  # Pieces of examples whose total size is < batch_size
  buf = []
  # Total size of examples in buf.
  buf_size = 0
  # Mask for full batches.
  full_mask = np.ones([hparams.batch_size], dtype=np.bool_)
  # Invariant: buf_size < batch_size.
  for dataset in datasets:
    if preprocessor is None:
      preprocessor = dataset.preprocessor
    elif dataset.preprocessor is not preprocessor:
      raise ValueError(
          'client_datasets should have the identical Preprocessor object, '
          f'got {preprocessor} vs {dataset.preprocessor}')
    if features is None:
      features = set(dataset.raw_examples)
    elif features != set(dataset.raw_examples):
      raise ValueError('client_datasets should have identical features, '
                       f'got {features} vs {list(dataset.raw_examples)}')
    size = len(dataset)
    examples = dataset.raw_examples
    if buf_size + size < hparams.batch_size:
      buf.append(examples)
      buf_size += size
      continue
    if buf:
      # Emit what's in buf.
      start = hparams.batch_size - buf_size
      buf.append(slice_examples(examples, slice(start)))
      yield attach_mask(preprocessor(concat_examples(buf)), full_mask)
      buf.clear()
      buf_size = 0
    else:
      start = 0
    # Emit batches in examples.
    while start + hparams.batch_size < size:
      yield attach_mask(
          preprocessor(
              slice_examples(examples, slice(start,
                                             start + hparams.batch_size))),
          full_mask)
      start += hparams.batch_size
    # Buffer remaining.
    if start < size:
      buf.append(slice_examples(examples, slice(start, size)))
      buf_size += size - start
  if buf:
    final_examples = preprocessor(concat_examples(buf))
    final_batch_size = _pick_final_batch_size(buf_size, hparams.batch_size,
                                              hparams.num_batch_size_buckets)
    yield pad_examples(final_examples, final_batch_size)


def buffered_shuffle(source: Iterable[Any], buffer_size: int,
                     rng: np.random.RandomState) -> Iterator[Any]:
  """Shuffles an iterable via buffered shuffling."""
  it = iter(source)
  buf = list(itertools.islice(it, buffer_size))
  rng.shuffle(buf)
  for i in it:
    r, buf[0] = buf[0], i
    swap = rng.randint(buffer_size)
    if swap < buffer_size - 1:
      buf[swap], buf[0] = buf[0], buf[swap]
    yield r
  for i in buf:
    yield i


def buffered_shuffle_batch_client_datasets(
    datasets: Iterable[ClientDataset], batch_size: int, buffer_size: int,
    rng: np.random.RandomState) -> Iterator[Examples]:
  """Shuffles and batches examples from multiple client datasets.

  This just makes 1 pass over the examples. To achieve repeated iterations,
  create an infinite shuffled stream of datasets first (e.g. using
  buffered_shuffle()).

  Args:
    datasets: ClientDatasets to be batched. All ClientDatasets must have the
      same Preprocessor object attached.
    batch_size: Desired batch size.
    buffer_size: Number of examples to buffer during shuffling.
    rng: Source of randomness.

  Yields:
    Batches of examples. For a finite stream of datasets, the final batch might
    be smaller than ``batch_size``.

  Raises:
    ValueError: If any 2 client datasets have different Preprocessors.
    ValueError: If any 2 client datasets have different features.
  """

  def gen_items():
    """Generates the processor, then pairs of (examples, index)."""
    preprocessor = None
    features = None
    for dataset in datasets:
      if preprocessor is None:
        preprocessor = dataset.preprocessor
        yield preprocessor
      elif dataset.preprocessor is not preprocessor:
        raise ValueError(
            'client_datasets should have the identical Preprocessor object, '
            f'got {preprocessor} vs {dataset.preprocessor}')
      if features is None:
        features = set(dataset.raw_examples)
      elif features != set(dataset.raw_examples):
        raise ValueError('client_datasets should have identical features, '
                         f'got {features} vs {list(dataset.raw_examples)}')
      for i in range(len(dataset)):
        yield (dataset.raw_examples, i)

  it = gen_items()
  try:
    preprocessor = next(it)
  except StopIteration:
    return
  buf = []
  for item in buffered_shuffle(it, buffer_size, rng):
    buf.append(item)
    if len(buf) == batch_size:
      yield preprocessor(
          concat_examples([slice_examples(e, slice(i, i + 1)) for e, i in buf]))
      buf.clear()
  if buf:
    yield preprocessor(
        concat_examples([slice_examples(e, slice(i, i + 1)) for e, i in buf]))


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


def concat_examples(many_examples: Iterable[Examples]) -> Examples:
  combined = collections.defaultdict(list)
  for examples in many_examples:
    for k, v in examples.items():
      combined[k].append(v)
  return {k: np.concatenate(v, axis=0) for k, v in combined.items()}


def attach_mask(examples: Examples, mask: np.ndarray) -> Examples:
  if EXAMPLE_MASK_KEY in examples:
    raise ValueError(
        f'mask key {EXAMPLE_MASK_KEY!r} is already present in examples {examples}'
    )
  return {**examples, EXAMPLE_MASK_KEY: mask}


def pad_examples(examples: Examples, size: int) -> Examples:
  """Pad examples to a fixed number of rows with 0 values.

  Args:
    examples: Examples to pad.
    size: The desired number of rows (i.e. the leading dimension size).

  Returns:
    Padded examples.

  Raises:
    ValueError: invalid inputs.
  """
  if EXAMPLE_MASK_KEY in examples:
    raise ValueError(
        f'mask key {EXAMPLE_MASK_KEY!r} is already present in examples {examples}'
    )
  current_size = num_examples(examples)
  if current_size > size:
    raise ValueError(f'Cannot pad {current_size} examples to size {size}')
  result = {EXAMPLE_MASK_KEY: np.arange(size) < current_size}
  for k, v in examples.items():
    padded = np.zeros((size,) + v.shape[1:], v.dtype)
    padded[:current_size] = v
    result[k] = padded
  return result
