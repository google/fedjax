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
"""A small library for working with evaluation metrics such as accuracy.

##  Quick overview

To evaluate model predictions, use a `Metric` object such as `Accuracy`. We
recommend `fedjax.experimental.model.evaluate_model()` in most scenarios, which
runs model prediction, and evaluation, on batches of N examples at a time for
greater computational efficiency.

```python
# Mock out Model.
model = fedjax.experimental.model.Model.new(
    init=lambda _: None,  # Unused.
    apply_for_train=lambda _, _, _: None,  # Unused.
    apply_for_eval=lambda _, batch: batch.get('pred'),
    train_loss=lambda _, _: None,  # Unused.
    eval_metrics{'accuracy': metrics.Accuracy()})
params = None  # Unused.
batches = [{'y': np.array([1, 0]), 'pred': np.array([[1.2, 0.4], [2.3, 0.1]])},
           {'y': np.array([1, 1]), 'pred': np.array([[0.3, 3.2], [2.1, 4.3]])}]
eval_results = fedjax.experimental.model.evaluate_model(model, params, batches)
print(eval_results)
# {'accuracy': 0.75}
```

A `Metric` object has 2 methods:

-   `zero()`: Returns an initial value for accumulating the statistic for this
metric.
-   `evaluate_example(example, prediction)`: Returns the statistic from
evaluating a single example, given the training `example` and the model
`prediction`.

Most `Metric`s follow the following convention for convenience:

-   `example` is a dict-like object from `str` to `jnp.ndarray`.
-   `prediction` is either a single `jnp.ndarray`, or a dict-like object from
`str` to `jnp.ndarray`.

Conceptually, we can also use a simple for loop to evaluate a collection of
examples and model predictions:

```python
# By default, the `Accuracy` metric treats `example['y']` as the true label, and
# `prediction` as a single `jnp.ndarray` of class scores.
metric = Accuracy()
stat = metric.zero()
# We are iterating over individual examples, not batches.
for example, prediction in [({'y': jnp.array(1)}, jnp.array([0., 1.])),
                            ({'y': jnp.array(0)}, jnp.array([1., 0.])),
                            ({'y': jnp.array(1)}, jnp.array([1., 0.])),
                            ({'y': jnp.array(0)}, jnp.array([2., 0.]))]:
  stat = stat.merge(metric.evaluate_example(example, prediction))
print(stat.result())
# 0.75
```

In practice, for greater computational efficiency, we run model prediction not
on a single example, but a batch of N examples at a time.
`fedjax.experimental.model.evaluate_model()` is provides a simple way to do so.
Under the hood, it calls `evaluate_batch()`.

```python
metric = Accuracy()
stat = metric.zero()
# We are iterating over batches.
for batch_example, batch_prediction in [
  ({'y': jnp.array([1, 0])}, jnp.array([[0., 1.], [1., 0.]])),
  ({'y': jnp.array([1, 0])}, jnp.array([[1., 0.], [2., 0.]]))]:
  stat = stat.merge(evaluate_batch(metric, batch_example, batch_prediction))
print(stat.result())
# 0.75
```

##  Under the hood

For most users, it is sufficient to know how to use existing `Metric` subclasses
such as `Accuracy` with `fedjax.experimental.model.evaluate_model()`. This
section is intended for those who would like to write new metrics.

### From algebraic structures to `Metric` and `Stat`

There are 2 abstraction in this library, `Metric` and `Stat`. Before going into
details of these classes, let's first consider a few abstract properties related
to evaluation metrics, using accuracy as an example.

When evaluating accuracy on a dataset, we wish to know the proportion of
examples that are correctly predicted by the model. Because a dataset might be
too large to fit into memory, we need to divide the work by partitioning the
dataset into smaller subsets, evaluate each separately, and finally somehow
combine the results. Assuming the subsets can be of different sizes, although
the accuracy value is a single number, we cannot just average the accuracy
values from each partition to arrive at the overall accuracy. Instead, we need 2
numbers from each subset:

-   The number of examples in this subset,
-   How many of them are correctly predicted.

We call these two numbers from each subset a _statistic_. The domain (the set of
possible values) of the statistic in the case of accuracy is `{(0, 0)} ∪ {(a,
b) | a >= 0, b > 0}`.

With the numbers of examples and correct predictions from 2 disjoint subsets, we
add the numbers up to get the number of examples and correct predictions for the
union of the 2 subsets. We call this operation from 2 statistics into 1 a
`merge` operation.

Let `f(S)` be the function that gives us the statistic from a subset of
examples. It is easy to see for two disjoint subsets `A` and `B`, `merge(f(A),
f(B))` should be equal to `f(A ∪ B)`. If no such `merge` exists, we cannot
evaluate the dataset by partitioning the work. This requirement alone implies
the domain of a statistic, and the `merge` operation forms a specific algebraic
structure (a commutative monoid).

-   `I := f(empty set)` is one and the only identity element w.r.t. `merge`
(i.e. `merge(I, x) == merge(x, I) == x`.
-   `merge()` is commutative and associative.

Further, we can see `f(S)` can be defined just knowing two types of values:

-   `f(empty set)`, i.e. `I`;
-   `f({x})` for any single example `x`.

For any other subset `S`, we can derive the value of `f(S)` using these values
and `merge`. `Metric` is simply the `f(S)` function above, defined in 2
corresponding parts:

-   `Metric.zero()` is `f(empty set)`.
-   `Metric.evaluate_example(example, prediction)` is `f({x})` for a single
example.

On the other hand, `Stat` stores a single statistic, a `merge` method for
combining two, and a `result` method for producing the final metric value.

To implement `Accuracy` as a subclass of `Metric`, we first need to know what
`Stat` to use. In this case, the statistic domain and `merge` is implemented by
a `MeanStat`. A `MeanStat` holds two values:

-   `accum` is the weighted sum of values, i.e. the number of correct
predictions in the case of accuracy.
-   `weight`is the sum of weights, i.e. the number of examples in the case of
accuracy.

`merge` adds up the respective `accum` and `weight` from two `MeanStat` objects.

Sometimes, a new `Stat` subclass is necessary. In that case, it is very
important to make sure the implementation has a clear definition of the domain,
and the `merge` operation adheres to the properties regarding identity element,
commutativity, and associativity (e.g. if we unknowingly allow pairs of `(x, 0)`
for `x != 0` into the domain of a `MeanStat`, `merge((x, 0), (a, b))` will
produce a statistic that leads to incorrect final metric values, i.e. `(a+x)/b`,
instead of `a/b`).

### Batching `Stat`s

In most cases, the final value of an evaluation is simply a scalar, and the
corresponding statistic is also a tuple of a few scalar values. However, for the
same reason why `jax.vmap` is a lot more efficient than a for loop, it is a lot
more efficient to store multiple `Stat` values as a `Stat` of arrays, instead of
a list of `Stat` objects. Thus instead of a list `[MeanStat(1, 2), MeanStat(3,
4), MeanStat(5, 6)]` (call these "rank 0" `Stat`s), we can batch the 3
statitstics as a single `MeanStat(jnp.array([1, 3, 5]), jnp.array([2, 4, 6]))`.
A `Stat` object holding a single statistic is a "rank 0" `Stat`. A `Stat` object
holding a vector of statistics is a "rank 1" `Stat`. Similarly, a `Stat` object
may also hold a matrix, a 3D array, etc, of statistics. These are higher rank
`Stat`s.

In the end, we want just 1 final metric value instead of a length 3 vector, or a
2x2 matrix, of metric values. To finally go back to a single statistic (), we
need to `merge` statistics stored in these arrays. Each `Stat` subclass provides
a `reduce` method to do just that. The combination of `jax.vmap` over
`Metric.evaluate_example`, and `Stat.reduce`, is how we get an efficient
`evaluate_batch()` function (of course, the real `evaluate_batch` is `jax.jit`'d
so that the same `jax.vmap` transformation etc does not need to happen over and
over.

Importantly, for a higher rank `Stat` (rank >= 1), only `reduce` needs to work.
Other operations such as `merge` and `result` only needs to support rank 0
`Stat`s (because we can get the batched version of these via `jax.vmap`).

```python
metric = Accuracy()
stat = metric.zero()
# We are iterating over batches.
for batch_example, batch_prediction in [
  ({'y': jnp.array([1, 0])}, jnp.array([[0., 1.], [1., 0.]])),
  ({'y': jnp.array([1, 0])}, jnp.array([[1., 0.], [2., 0.]]))]:
  # Get a batch of statistics as a single Stat object.
  batch_stat = jax.vmap(metric.evaluate_example)(batch_example,
  batch_prediction)
  # Merge the reduced single statistic onto the accumulator.
  stat = stat.merge(batch_stat.reduce())
print(stat.result())
# 0.75
```

Being able to batch `Stat`s also allow us to do other interesting things, for
example,
-   `evaluate_batch` accepts an optional per-example `mask` so it can work on
padded batches.
-   We can define a `PerDomain` metric for any base metric so that we can get
accuracy where examples are partitioned by a domain id.

### Creating a new metric

Most likely, a new metric will just return a `MeanStat` or a `SumStat`. If
that's the case, simply following implement a new `Metric` following the
guidelines in `Metric`'s class docstring.

If a new `Stat` is necessary, following the guidelines in `Stat`'s docstring.
"""

import abc
import functools
from typing import Optional, Tuple

from fedjax import core
from fedjax.experimental.typing import BatchExample
from fedjax.experimental.typing import BatchPrediction
from fedjax.experimental.typing import SingleExample
from fedjax.experimental.typing import SinglePrediction

import jax
import jax.numpy as jnp


class Stat(metaclass=abc.ABCMeta):
  """Stat keeps some statistic, along with operations over them.

  Most users will only need to interact with a `Stat` object via `result()`.

  For those who need to create new metrics, please first read the "Under the
  hood" section of the module docstring.

  Most `Stat`'s domain (the set of possible statistic values) has constraints,
  it is thus usually a good practice to offer and use factory methods to
  construct new `Stat` objects instead of directly assigning the fields.

  To work with various jax constructs, a concrete `Stat` should be a PyTree.
  This is easily achieved with `fedjax.dataclass`.

  A `Stat` may hold either a single statistic (a rank 0 `Stat`), or an array of
  statistics (a higher rank `Stat`). `result` and `merge` only needs to work on
  a rank 0 `Stat`. `reduce` only needs to work on a higher rank `Stat`.
  """

  @abc.abstractmethod
  def result(self) -> jnp.ndarray:
    """Calculates the metric value from the statistic value.

    For example, `MeanStat.result()` calculates a weighted average.

    Returns:
      The return value of `result()` must be a `jnp.ndarray`.
    """

  @abc.abstractmethod
  def merge(self, other: 'Stat') -> 'Stat':
    """Merges two `Stat` objects into a new `Stat` with merged statistics.

    Args:
      other: Another `Stat` object of the same type.

    Returns:
      A new `Stat` object of the same type with merged statistics.
    """

  @abc.abstractmethod
  def reduce(self, axis: Optional[int] = 0):
    """Reduces a higher rank statistic along a given `axis`.

    See the class docstring for details.

    Args:
      axis: An integer axis index, or `None`.

    Returns:
      A new `Stat` object of the same type.
    """

  def __str__(self) -> str:
    return f'{repr(self)} => {self.result()}'


@core.dataclass
class MeanStat(Stat):
  """Statistic for weighted mean calculation.

  A `MeanStat` maintains the weighted sum (`accum`) and the sum of weights
  (`weight`) for calculating the weighted mean.

  Prefer using the `new` or `single` factory methods instead of directly
  assigning to fields.
  """
  accum: jnp.ndarray
  weight: jnp.ndarray

  @classmethod
  def new(cls, accum, weight) -> 'MeanStat':
    """Creates a sanitized `MeanStat`.

    The domain of a weighted mean statistic is:

    ```
    {(0, 0)} ∪ {(a, b) | a >= 0, b > 0}
    ```

    `new()` sanitizes values outside the domain into the identity (zeros).

    Args:
      accum: A value convertible to `jnp.ndarray`.
      weight: A value convertible to `jnp.ndarray`.

    Returns:
      The sanitized `MeanStat`.
    """
    weight = jnp.maximum(0, jnp.array(weight, copy=False))
    accum = jnp.where(weight == 0, 0, jnp.array(accum, copy=False))
    return cls(accum, weight)

  def result(self) -> jnp.ndarray:
    return jnp.where(self.weight == 0, 0, self.accum / self.weight)

  def merge(self, other: 'MeanStat') -> 'MeanStat':
    accum = self.accum + other.accum
    weight = self.weight + other.weight
    return MeanStat.new(accum, weight)

  def reduce(self, axis: Optional[int] = 0):
    return MeanStat.new(
        jnp.sum(self.accum, axis=axis), jnp.sum(self.weight, axis=axis))


@core.dataclass
class SumStat(Stat):
  """Statistic for summing values.

  A `SumStat` simply maintains the sum of values (`accum`).
  """
  accum: jnp.ndarray

  @classmethod
  def new(cls, accum: jnp.ndarray) -> 'SumStat':
    return cls(jnp.array(accum, copy=False))

  def result(self) -> jnp.ndarray:
    return self.accum

  def merge(self, other: 'SumStat') -> 'SumStat':
    return SumStat.new(self.accum + other.accum)

  def reduce(self, axis: Optional[int] = 0):
    return SumStat.new(jnp.sum(self.accum, axis=axis))


class Metric(metaclass=abc.ABCMeta):
  """`Metric` is the conceptual metric (like accuracy).

  It defines two methods:

  -   `evaluate_example()` evaluates a single example, and returns a `Stat`
  object.
  -   `zero()` returns the identity value for what `evaluate_example()` returns.

  Given a `Metric` object `m`, let

  -   `u = m.zero()`
  -   `v = m.evaluate_example(...)`

  We require that

  -   `type(u) == type(v)`.
  -   `u.merge(v) == v.merge(u) == v`.
  -   Components of `u` has the same shape as the counter parts in `v`.
  """

  @abc.abstractmethod
  def zero(self) -> Stat:
    """Returns a `Stat` such that merging with it is an identity operation.

    e.g. for accuracy: `MeanStat.new(0., 0.)`

    Returns:
      `Stat` identity value.
    """

  @abc.abstractmethod
  def evaluate_example(self, example: SingleExample,
                       prediction: SinglePrediction) -> Stat:
    """Evaluates a single example.

    e.g. for accuracy: `MeanStat.new(num_correct, num_total)`

    Args:
      example: A single input example (e.g. one sentence for language).
      prediction: Output for `example` from `Model.apply_for_eval`.

    Returns:
      `Stat` value.
    """


def apply_mask(mask: jnp.ndarray, a: jnp.ndarray,
               b: jnp.ndarray) -> jnp.ndarray:
  """Applies mask on the leading dimension."""
  rank = max(len(a.shape), len(b.shape))
  return jnp.where(jnp.expand_dims(mask, tuple(range(1, rank))), a, b)


@functools.partial(jax.jit, static_argnums=0)
def evaluate_batch(metric: Metric,
                   batch_example: BatchExample,
                   batch_prediction: BatchPrediction,
                   batch_mask: Optional[jnp.ndarray] = None):
  """Evaluates a batch using a metric."""
  batch_stat = jax.vmap(metric.evaluate_example)(batch_example,
                                                 batch_prediction)
  if batch_mask is not None:
    batch_stat = jax.tree_util.tree_multimap(
        functools.partial(apply_mask, batch_mask), batch_stat, metric.zero())
  return batch_stat.reduce()


def unreduced_cross_entropy_loss(targets: jnp.ndarray,
                                 preds: jnp.ndarray) -> jnp.ndarray:
  """Returns unreduced cross entropy loss."""
  num_classes = preds.shape[-1]
  log_preds = jax.nn.log_softmax(preds)
  one_hot_targets = jax.nn.one_hot(targets, num_classes)
  return -jnp.sum(one_hot_targets * log_preds, axis=-1)


@core.dataclass
class CrossEntropyLoss(Metric):
  """Metric for cross entropy loss.

  Attributes:
    target_key: Key name in `example` for target.
    pred_key: Key name in `prediction` for unnormalized model output pred.
  """
  target_key: str = 'y'
  pred_key: Optional[str] = None

  def zero(self) -> MeanStat:
    return MeanStat.new(0., 0.)

  def evaluate_example(self, example: SingleExample,
                       prediction: SinglePrediction) -> MeanStat:
    """Computes cross entropy loss for a single example.

    Args:
      example: One example with target in range [0, num_classes) of shape [1].
      prediction: Unnormalized prediction for `example` of shape [num_classes].

    Returns:
      `MeanStat` for loss for a single example.
    """
    target = example[self.target_key]
    pred = prediction if self.pred_key is None else prediction[self.pred_key]
    loss = unreduced_cross_entropy_loss(target, pred)
    return MeanStat.new(loss, 1.)


@core.dataclass
class Accuracy(Metric):
  """Metric for accuracy.

  Attributes:
    target_key: Key name in `example` for target.
    pred_key: Key name in `prediction` for unnormalized model output pred.
  """
  target_key: str = 'y'
  pred_key: Optional[str] = None

  def zero(self) -> MeanStat:
    return MeanStat.new(0., 0.)

  def evaluate_example(self, example: SingleExample,
                       prediction: SinglePrediction) -> MeanStat:
    """Computes accuracy for a single example.

    Args:
      example: One example with target in range [0, num_classes) of shape [1].
      prediction: Unnormalized prediction for `example` of shape [num_classes].

    Returns:
      `MeanStat` for accuracy for a single example.
    """
    target = example[self.target_key]
    pred = prediction if self.pred_key is None else prediction[self.pred_key]
    correct = (target == jnp.argmax(pred, axis=-1)).astype(jnp.float32)
    return MeanStat.new(correct, 1.)


def _target_weight(
    target: jnp.ndarray, masked_target_values: Tuple[jnp.ndarray,
                                                     ...]) -> jnp.ndarray:
  target_weight = jnp.ones_like(target, dtype=jnp.float32)
  for mv in masked_target_values:
    target_weight *= (target != mv)
  return target_weight


@core.dataclass
class SequenceTokenCrossEntropyLoss(Metric):
  """Metric for token cross entropy loss for a sequence example.

  Attributes:
    target_key: Key name in `example` for target.
    pred_key: Key name in `prediction` for unnormalized model output pred.
    masked_target_values: Target values that should be ignored in computation.
      This is typically used to ignore padding values in computation.
  """
  target_key: str = 'y'
  pred_key: Optional[str] = None
  masked_target_values: Tuple[jnp.ndarray, ...] = (0,)

  def zero(self) -> MeanStat:
    return MeanStat.new(0., 0.)

  def evaluate_example(self, example: SingleExample,
                       prediction: SinglePrediction) -> MeanStat:
    """Computes token cross entropy loss for a single sequence example.

    Args:
      example: One example with target in range [0, num_classes) of shape [1].
      prediction: Unnormalized prediction for `example` of shape [num_classes].

    Returns:
      `MeanStat` for token loss for a single sequence example.
    """
    target = example[self.target_key]
    pred = prediction if self.pred_key is None else prediction[self.pred_key]
    target_weight = _target_weight(target, self.masked_target_values)
    token_loss = unreduced_cross_entropy_loss(target, pred)
    return MeanStat.new(
        jnp.sum(token_loss * target_weight), jnp.sum(target_weight))


@core.dataclass
class SequenceCrossEntropyLoss(Metric):
  """Metric for total cross entropy loss for a sequence example.

  Attributes:
    target_key: Key name in `example` for target.
    pred_key: Key name in `prediction` for unnormalized model output pred.
    masked_target_values: Target values that should be ignored in computation.
      This is typically used to ignore padding values in computation.
  """
  target_key: str = 'y'
  pred_key: Optional[str] = None
  masked_target_values: Tuple[jnp.ndarray, ...] = (0,)

  def zero(self) -> MeanStat:
    return MeanStat.new(0., 0.)

  def evaluate_example(self, example: SingleExample,
                       prediction: SinglePrediction) -> MeanStat:
    """Computes total cross entropy loss for a single sequence example.

    Args:
      example: One example with target in range [0, num_classes) of shape [1].
      prediction: Unnormalized prediction for `example` of shape [num_classes].

    Returns:
      `MeanStat` for total loss for a single sequence example.
    """
    target = example[self.target_key]
    pred = prediction if self.pred_key is None else prediction[self.pred_key]
    target_weight = _target_weight(target, self.masked_target_values)
    token_loss = unreduced_cross_entropy_loss(target, pred)
    # Change weight from number of non masked target tokens to 1 if the sequence
    # contains any non masked tokens or 0 if the entire sequence is masked.
    return MeanStat.new(
        jnp.sum(token_loss * target_weight), jnp.any(target_weight))


@core.dataclass
class SequenceTokenAccuracy(Metric):
  """Metric for token accuracy for a sequence example.

  Attributes:
    target_key: Key name in `example` for target.
    pred_key: Key name in `prediction` for unnormalized model output pred.
    masked_target_values: Target values that should be ignored in computation.
      This is typically used to ignore padding values in computation.
  """
  target_key: str = 'y'
  pred_key: Optional[str] = None
  masked_target_values: Tuple[jnp.ndarray, ...] = (0,)

  def zero(self) -> MeanStat:
    return MeanStat.new(0., 0.)

  def evaluate_example(self, example: SingleExample,
                       prediction: SinglePrediction) -> MeanStat:
    """Computes token accuracy for a single sequence example.

    Args:
      example: One example with target in range [0, num_classes) of shape [1].
      prediction: Unnormalized prediction for `example` of shape [num_classes].

    Returns:
      `MeanStat` for token accuracy for a single sequence example.
    """
    target = example[self.target_key]
    pred = prediction if self.pred_key is None else prediction[self.pred_key]
    target_weight = _target_weight(target, self.masked_target_values)
    correct = (target == jnp.argmax(pred, axis=-1)).astype(jnp.float32)
    return MeanStat.new(
        jnp.sum(correct * target_weight), jnp.sum(target_weight))


@core.dataclass
class SequenceTokenCount(Metric):
  """Metric for count of non masked tokens for a sequence example.

  Attributes:
    target_key: Key name in `example` for target.
    masked_target_values: Target values that should be ignored in computation.
      This is typically used to ignore padding values in computation.
  """
  target_key: str = 'y'
  masked_target_values: Tuple[jnp.ndarray, ...] = (0,)

  def zero(self) -> SumStat:
    return SumStat.new(0.)

  def evaluate_example(self, example: SingleExample,
                       prediction: SinglePrediction) -> SumStat:
    """Computes total number of non masked tokens in a single sequence example.

    Args:
      example: One example with target in range [0, num_classes) of shape [1].
      prediction: Unnormalized prediction for `example` of shape [num_classes].

    Returns:
      `SumStat` for count of non masked tokens for a single sequence example.
    """
    del prediction
    target = example[self.target_key]
    target_weight = _target_weight(target, self.masked_target_values)
    return SumStat.new(jnp.sum(target_weight))


@core.dataclass
class SequenceTruncationRate(Metric):
  """Metric for truncation rate for a sequence example.

  Attributes:
    eos_target_value: Target value denoting end of sequence. Truncated sequences
      will not have this value.
    target_key: Key name in `example` for target.
    masked_target_values: Target values that should be ignored in computation.
      This is typically used to ignore padding values in computation.
  """
  eos_target_value: jnp.ndarray
  target_key: str = 'y'
  masked_target_values: Tuple[jnp.ndarray, ...] = (0,)

  def zero(self) -> MeanStat:
    return MeanStat.new(0., 0.)

  def evaluate_example(self, example: SingleExample,
                       prediction: SinglePrediction) -> MeanStat:
    """Computes truncation rate for a single sequence example.

    Args:
      example: One example with target in range [0, num_classes) of shape [1].
      prediction: Unnormalized prediction for `example` of shape [num_classes].

    Returns:
      `MeanStat` for truncation rate for a single sequence.
    """
    del prediction
    target = example[self.target_key]
    target_weight = _target_weight(target, self.masked_target_values)
    not_empty = jnp.sum(jnp.any(target_weight))
    target_is_truncated = jnp.all(target != self.eos_target_value)
    return MeanStat.new(target_is_truncated * not_empty, not_empty)


@core.dataclass
class SequenceTokenOOVRate(Metric):
  """Metric for out-of-vocabulary (OOV) rate for a sequence example.

  Attributes:
    oov_target_values: Target values denoting out-of-vocabulary values.
    target_key: Key name in `example` for target.
    masked_target_values: Target values that should be ignored in computation.
      This is typically used to ignore padding values in computation.
  """
  oov_target_values: Tuple[jnp.ndarray, ...]
  target_key: str = 'y'
  masked_target_values: Tuple[jnp.ndarray, ...] = (0,)

  def zero(self) -> MeanStat:
    return MeanStat.new(0., 0.)

  def evaluate_example(self, example: SingleExample,
                       prediction: SinglePrediction) -> MeanStat:
    """Computes token out of vocabulary rate for a single sequence example.

    Args:
      example: One example with target in range [0, num_classes) of shape [1].
      prediction: Unnormalized prediction for `example` of shape [num_classes].

    Returns:
      `MeanStat` for token out of vocabulary rate for a single sequence.
    """
    del prediction
    target = example[self.target_key]
    target_weight = _target_weight(target, self.masked_target_values)
    target_oov = jnp.ones_like(target, dtype=jnp.float32)
    for oov_value in self.oov_target_values:
      target_oov *= (target == oov_value)
    return MeanStat.new(
        jnp.sum(target_oov * target_weight), jnp.sum(target_weight))


@core.dataclass
class SequenceLength(Metric):
  """Metric for length for a sequence example.

  Attributes:
    target_key: Key name in `example` for target.
    masked_target_values: Target values that should be ignored in computation.
      This is typically used to ignore padding values in computation.
  """
  target_key: str = 'y'
  masked_target_values: Tuple[jnp.ndarray, ...] = (0,)

  def zero(self) -> MeanStat:
    return MeanStat.new(0., 0.)

  def evaluate_example(self, example: SingleExample,
                       prediction: SinglePrediction) -> MeanStat:
    """Computes non masked length for a single sequence example.

    Args:
      example: One example with target in range [0, num_classes) of shape [1].
      prediction: Unnormalized prediction for `example` of shape [num_classes].

    Returns:
      `MeanStat` for non masked length for a single sequence.
    """
    del prediction
    target = example[self.target_key]
    target_weight = _target_weight(target, self.masked_target_values)
    return MeanStat.new(jnp.sum(target_weight), jnp.sum(jnp.any(target_weight)))
