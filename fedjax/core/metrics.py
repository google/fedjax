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
"""A small library for working with evaluation metrics such as accuracy."""

import abc
import functools
from typing import Optional, Tuple

from fedjax.core import dataclasses
from fedjax.core import util
from fedjax.core.typing import BatchExample
from fedjax.core.typing import BatchPrediction
from fedjax.core.typing import SingleExample
from fedjax.core.typing import SinglePrediction

import jax
import jax.numpy as jnp


class Stat(metaclass=abc.ABCMeta):
  """Stat keeps some statistic, along with operations over them.

  Most users will only need to interact with a :class:`Stat` object via
  :py:meth:`~Stat.result`

  For those who need to create new metrics, please first read the
  :ref:`under-the-hood` section of the module docstring.

  Most :class:`Stat`'s domain (the set of possible statistic values) has
  constraints, it is thus usually a good practice to offer and use factory
  methods to construct new :class:`Stat` objects instead of directly assigning
  the fields.

  To work with various jax constructs, a concrete :class:`Stat` should be a
  PyTree.
  This is easily achieved with ``fedjax.dataclass``.

  A :class:`Stat` may hold either a single statistic (a rank 0 :class:`Stat`),
  or an array of statistics (a higher rank :class:`Stat`).
  :py:meth:`~Stat.result` and :py:meth:`~Stat.merge` only needs to
  work on a rank 0 :class:`Stat` :py:meth:`~Stat.reduce` only needs to work on
  a higher rank :class:`Stat`
  """

  @abc.abstractmethod
  def result(self) -> jnp.ndarray:
    """Calculates the metric value from the statistic value.

    For example, :meth:`MeanStat.result` calculates a weighted average.

    Returns:
      The return value must be a ``jnp.ndarray``.
    """

  @abc.abstractmethod
  def merge(self, other: 'Stat') -> 'Stat':
    """Merges two Stat objects into a new Stat with merged statistics.

    Args:
      other: Another Stat object of the same type.

    Returns:
      A new Stat object of the same type with merged statistics.
    """

  @abc.abstractmethod
  def reduce(self, axis: Optional[int] = 0) -> 'Stat':
    """Reduces a higher rank statistic along a given ``axis``.

    See the class docstring for details.

    Args:
      axis: An integer axis index, or ``None``.

    Returns:
      A new Stat object of the same type.
    """

  def __str__(self) -> str:
    return f'{repr(self)} => {self.result()}'


@dataclasses.dataclass
class MeanStat(Stat):
  """Statistic for weighted mean calculation.

  Prefer using the :meth:`MeanStat.new()` factory method instead of directly
  assigning to fields.

  Example::

    stat_0 = MeanStat.new(accum=1, weight=2)
    stat_1 = MeanStat.new(accum=2, weight=3)
    merged_stat = stat_0.merge(stat_1)
    print(merged_stat)
    # MeanState(accum=3, weight=5) => 0.6

    stat = MeanStat.new(jnp.array([1, 2, 4]), jnp.array([1, 1, 0]))
    reduced_stat = stat.reduce()
    print(reduced_stat)
    # MeanStat(accum=3, weight=2) => 1.5

  Attributes:
    accum: The weighted sum.
    weight: The sum of weights.
  """
  accum: jnp.ndarray
  weight: jnp.ndarray

  @classmethod
  def new(cls, accum, weight) -> 'MeanStat':
    """Creates a sanitized MeanStat.

    The domain of a weighted mean statistic is:

    .. math::

      \{(0, 0)\} ∪ \{(a, b) | a >= 0, b > 0\}


    new() sanitizes values outside the domain into the identity (zeros).

    Args:
      accum: A value convertible to ``jnp.ndarray``.
      weight: A value convertible to ``jnp.ndarray``.

    Returns:
      The sanitized MeanStat.
    """
    weight = jnp.maximum(0, jnp.array(weight, copy=False))
    accum = jnp.where(weight == 0, 0, jnp.array(accum, copy=False))
    return cls(accum, weight)

  def result(self) -> jnp.ndarray:
    return util.safe_div(self.accum, self.weight)

  def merge(self, other: 'MeanStat') -> 'MeanStat':
    accum = self.accum + other.accum
    weight = self.weight + other.weight
    return MeanStat.new(accum, weight)

  def reduce(self, axis: Optional[int] = 0) -> 'MeanStat':
    return MeanStat.new(
        jnp.sum(self.accum, axis=axis), jnp.sum(self.weight, axis=axis))


@dataclasses.dataclass
class SumStat(Stat):
  """Statistic for summing values.

  Example::

    stat_0 = SumStat.new(accum=1)
    stat_1 = SumStat.new(accum=2)
    merged_stat = stat_0.merge(stat_1)
    print(merged_stat)
    # SumStat(accum=3) => 3

    stat = SumStat.new(jnp.array([1, 2, 1]))
    reduced_stat = stat.reduce()
    print(reduced_stat)
    # SumStat(accum=4) => 4

  Attributes:
    accum: Sum of values.
  """
  accum: jnp.ndarray

  @classmethod
  def new(cls, accum: jnp.ndarray) -> 'SumStat':
    """Creates a sanitized SumStat."""
    return cls(jnp.array(accum, copy=False))

  def result(self) -> jnp.ndarray:
    return self.accum

  def merge(self, other: 'SumStat') -> 'SumStat':
    return SumStat.new(self.accum + other.accum)

  def reduce(self, axis: Optional[int] = 0) -> 'SumStat':
    return SumStat.new(jnp.sum(self.accum, axis=axis))


class Metric(metaclass=abc.ABCMeta):
  """Metric is the conceptual metric (like accuracy).

  It defines two methods:

  - :meth:`~Metric.evaluate_example` evaluates a single example, and returns a
    :class:`Stat` object.
  - :meth:`~Metric.zero` returns the identity value for what
    :meth:`~Metric.evaluate_example` returns.

  Given a :class:`Metric` object ``m``, let

  - ``u = m.zero()``
  - ``v = m.evaluate_example(...)``

  We require that

  - ``type(u) == type(v)``.
  - ``u.merge(v) == v.merge(u) == v``.
  - Components of ``u`` has the same shape as the counter parts in ``v``.
  """

  @abc.abstractmethod
  def zero(self) -> Stat:
    """Returns a Stat such that merging with it is an identity operation.

    e.g. for accuracy: ``MeanStat.new(0., 0.)``

    Returns:
      Stat identity value.
    """

  @abc.abstractmethod
  def evaluate_example(self, example: SingleExample,
                       prediction: SinglePrediction) -> Stat:
    """Evaluates a single example.

    e.g. for accuracy: ``MeanStat.new(num_correct, num_total)``

    Args:
      example: A single input example (e.g. one sentence for language).
      prediction: Output for ``example`` from
        :meth:`fedjax.core.models.Model.apply_for_eval`.

    Returns:
      Stat value.
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
                   batch_mask: Optional[jnp.ndarray] = None) -> Stat:
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


@dataclasses.dataclass
class CrossEntropyLoss(Metric):
  """Metric for cross entropy loss.

  Example::

    example = {'y': jnp.array(1)}
    prediction = jnp.array([1.2, 0.4])
    metric = CrossEntropyLoss()
    print(metric.evaluate_example(example, prediction))
    # MeanStat(accum=1.1711007, weight=1) => 1.1711007

  Attributes:
    target_key: Key name in ``example`` for target.
    pred_key: Key name in ``prediction`` for unnormalized model output pred.
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
      prediction: Unnormalized prediction for ``example`` of shape [num_classes]

    Returns:
      MeanStat for loss for a single example.
    """
    target = example[self.target_key]
    pred = prediction if self.pred_key is None else prediction[self.pred_key]
    loss = unreduced_cross_entropy_loss(target, pred)
    return MeanStat.new(loss, 1.)


@dataclasses.dataclass
class Accuracy(Metric):
  """Metric for accuracy.

  Example::

    example = {'y': jnp.array(2)}
    prediction = jnp.array([0, 0, 1])
    metric = Accuracy()
    print(metric.evaluate_example(example, prediction))
    # MeanStat(accum=1, weight=1) => 1

  Attributes:
    target_key: Key name in ``example`` for target.
    pred_key: Key name in ``prediction`` for unnormalized model output pred.
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
      prediction: Unnormalized prediction for ``example`` of shape [num_classes]

    Returns:
      MeanStat for accuracy for a single example.
    """
    target = example[self.target_key]
    pred = prediction if self.pred_key is None else prediction[self.pred_key]
    correct = (target == jnp.argmax(pred, axis=-1)).astype(jnp.float32)
    return MeanStat.new(correct, 1.)


def get_target_weight(target: jnp.ndarray,
                      masked_target_values: Tuple[int, ...]) -> jnp.ndarray:
  target_weight = jnp.ones_like(target, dtype=jnp.float32)
  for mv in masked_target_values:
    target_weight *= (target != mv)
  return target_weight


@dataclasses.dataclass
class TopKAccuracy(Metric):
  """Metric for top k accuracy.

  This metric computes the number of times where the correct class
  is among the top k classes predicted.

  Example: top 3 accuracy

  - Dog => [Dog, Cat, Bird, Mouse, Penguin] ✓
  - Cat => [Bird, Mouse, Cat, Penguin, Dog] ✓
  - Dog => [Dog, Cat, Bird, Penguin, Mouse] ✓
  - Bird => [Bird, Cat, Mouse, Penguin, Dog] ✓
  - Cat => [Cat, Bird, Mouse, Dog, Penguin] ✓
  - Cat => [Cat, Mouse, Dog, Penguin, Bird] ✓
  - Mouse => [Penguin, Cat, Dog, Mouse, Bird] x
  - Penguin => [Dog, Mouse, Cat, Penguin, Bird] x

  6 correct predictions in top 3 predicted classes / 8 total examples
  = .75 top 3 accuracy

  Top k accuracy, also known as top n accuracy,
  is a useful metric when it comes to recommendations.
  One example would be the word recommendations on a virtual keyboard
  where three suggested words are displayed.

  For k=1, we strongly recommend using :class:`Accuracy` to avoid an
  unnecessary argsort. k < 1 will return 0. and k >= num_classes will
  return 1.

  If two or more classes have the same prediction, the classes will be
  considered in order of lowest to highest indices.

  Example::

    example = {'y': jnp.array(2)}
    prediction = jnp.array([0, 0.5, 0.2])
    metric = TopKAccuracy(k=2)
    print(metric.evaluate_example(example, prediction))
    # MeanStat(accum=1, weight=1) => 1

  Attributes:
    k: Number of top elements to look at for computing accuracy.
    target_key: Key name in ``example`` for target.
    pred_key: Key name in ``prediction`` for unnormalized model output pred.
  """
  k: int
  target_key: str = 'y'
  pred_key: Optional[str] = None

  def zero(self) -> Stat:
    return MeanStat.new(0., 0.)

  def evaluate_example(self, example: SingleExample,
                       prediction: SinglePrediction) -> Stat:

    """Computes top k accuracy for a single example.

    Args:
      example: One example with target in range [0, num_classes) of shape [1].
      prediction: Unnormalized prediction for ``example`` of shape [num_clases].

    Returns:
      MeanStat for top k accuracy for a single example.
    """
    target = example[self.target_key]
    pred = prediction if self.pred_key is None else prediction[self.pred_key]
    top_k_pred = jnp.argsort(-pred)[:self.k]
    correct = jnp.any(top_k_pred == target).astype(jnp.float32)
    return MeanStat.new(correct, 1.)


@dataclasses.dataclass
class SequenceTokenCrossEntropyLoss(Metric):
  """Metric for token cross entropy loss for a sequence example.

  Example::

    example = {'y': jnp.array([1, 0, 1])}
    prediction = jnp.array([[1.2, 0.4], [2.3, 0.1], [0.3, 3.2]])
    metric = SequenceTokenCrossEntropyLoss()
    print(metric.evaluate_example(example, prediction))
    # MeanStat(accum=1.2246635, weight=2) => 0.61233175

    per_position_metric = SequenceTokenCrossEntropyLoss(per_position=True)
    print(per_position_metric.evaluate_example(example, prediction))
    # MeanStat(accum=[1.1711007, 0., 0.05356275], weight=[1., 0., 1.]) => [1.1711007, 0., 0.05356275]

  Attributes:
    target_key: Key name in ``example`` for target.
    pred_key: Key name in ``prediction`` for unnormalized model output pred.
    masked_target_values: Target values that should be ignored in computation.
      This is typically used to ignore padding values in computation.
    per_position: Whether to keep output statistic per position or sum across
      positions for the entire sequence.
  """
  target_key: str = 'y'
  pred_key: Optional[str] = None
  masked_target_values: Tuple[int, ...] = (0,)
  per_position: bool = False

  def zero(self) -> MeanStat:
    return MeanStat.new(0., 0.)

  def evaluate_example(self, example: SingleExample,
                       prediction: SinglePrediction) -> MeanStat:
    """Computes token cross entropy loss for a single sequence example.

    Args:
      example: One example with target in range [0, num_classes) of shape [1].
      prediction: Unnormalized prediction for ``example`` of shape [num_classes]

    Returns:
      MeanStat for token loss for either a single sequence example or at each
        token position if ``per_position`` is ``True``.
    """
    target = example[self.target_key]
    pred = prediction if self.pred_key is None else prediction[self.pred_key]
    target_weight = get_target_weight(target, self.masked_target_values)
    token_loss = unreduced_cross_entropy_loss(target, pred)
    if self.per_position:
      return MeanStat.new(token_loss * target_weight, target_weight)
    return MeanStat.new(
        jnp.sum(token_loss * target_weight), jnp.sum(target_weight))


@dataclasses.dataclass
class SequenceCrossEntropyLoss(Metric):
  """Metric for total cross entropy loss for a sequence example.

  Example::

    example = {'y': jnp.array([1, 0, 1])}
    prediction = jnp.array([[1.2, 0.4], [2.3, 0.1], [0.3, 3.2]])
    metric = SequenceCrossEntropyLoss()
    print(metric.evaluate_example(example, prediction))
    # MeanStat(accum=1.2246635, weight=1) => 1.2246635

  Attributes:
    target_key: Key name in ``example`` for target.
    pred_key: Key name in ``prediction`` for unnormalized model output pred.
    masked_target_values: Target values that should be ignored in computation.
      This is typically used to ignore padding values in computation.
  """
  target_key: str = 'y'
  pred_key: Optional[str] = None
  masked_target_values: Tuple[int, ...] = (0,)

  def zero(self) -> MeanStat:
    return MeanStat.new(0., 0.)

  def evaluate_example(self, example: SingleExample,
                       prediction: SinglePrediction) -> MeanStat:
    """Computes total cross entropy loss for a single sequence example.

    Args:
      example: One example with target in range [0, num_classes) of shape [1].
      prediction: Unnormalized prediction for ``example`` of shape [num_classes]

    Returns:
      MeanStat for total loss for a single sequence example.
    """
    target = example[self.target_key]
    pred = prediction if self.pred_key is None else prediction[self.pred_key]
    target_weight = get_target_weight(target, self.masked_target_values)
    token_loss = unreduced_cross_entropy_loss(target, pred)
    # Change weight from number of non masked target tokens to 1 if the sequence
    # contains any non masked tokens or 0 if the entire sequence is masked.
    return MeanStat.new(
        jnp.sum(token_loss * target_weight), jnp.any(target_weight))


@dataclasses.dataclass
class SequenceTokenAccuracy(Metric):
  """Metric for token accuracy for a sequence example.

  Example::

    example = {'y': jnp.array([1, 2, 2, 1, 3, 0])}
    # prediction = [1, 0, 2, 1, 3, 0].
    prediction = jnp.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0],
                            [0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
    logits_mask = (0., 0., 0., jnp.NINF)
    metric = SequenceTokenAccuracy(logits_mask=logits_mask)
    print(metric.evaluate_example(example, prediction))
    # MeanStat(accum=3, weight=5) => 0.6

    per_position_metric = SequenceTokenAccuracy(logits_mask=logits_mask, per_position=True)
    print(per_position_metric.evaluate_example(example, prediction))
    # MeanStat(accum=[1., 0., 1., 1., 0., 0.], weight=[1., 1., 1., 1., 1., 0.]) => [1., 0., 1., 1., 0., 0.]

  Attributes:
    target_key: Key name in ``example`` for target.
    pred_key: Key name in ``prediction`` for unnormalized model output pred.
    masked_target_values: Target values that should be ignored in computation.
      This is typically used to ignore padding values in computation.
    logits_mask: Mask of shape [num_classes] to be applied for preds. This is
      typically used to discount predictions for out-of-vocabulary tokens.
    per_position: Whether to keep output statistic per position or sum across
      positions for the entire sequence.
  """
  target_key: str = 'y'
  pred_key: Optional[str] = None
  masked_target_values: Tuple[int, ...] = (0,)
  # logits_mask cannot be a jnp.ndarray nor np.ndarray because they are not
  # hashable and `Metric`s must be hashable for `evaluate_model`.
  logits_mask: Optional[Tuple[float, ...]] = None
  per_position: bool = False

  def zero(self) -> MeanStat:
    return MeanStat.new(0., 0.)

  def evaluate_example(self, example: SingleExample,
                       prediction: SinglePrediction) -> MeanStat:
    """Computes token accuracy for a single sequence example.

    Args:
      example: One example with target in range [0, num_classes) of shape [1].
      prediction: Unnormalized prediction for ``example`` of shape [num_classes]

    Returns:
      MeanStat for token accuracy for a single sequence example or at each
        token position if ``per_position`` is ``True``.
    """
    target = example[self.target_key]
    pred = prediction if self.pred_key is None else prediction[self.pred_key]
    if self.logits_mask is not None:
      logits_mask = jnp.array(self.logits_mask)
      pred += logits_mask
    target_weight = get_target_weight(target, self.masked_target_values)
    correct = (target == jnp.argmax(pred, axis=-1)).astype(jnp.float32)
    if self.per_position:
      return MeanStat.new(correct * target_weight, target_weight)
    return MeanStat.new(
        jnp.sum(correct * target_weight), jnp.sum(target_weight))


@dataclasses.dataclass
class SequenceTokenTopKAccuracy(Metric):
  """Metric for token top k accuracy for a sequence example.

  For more information on the top k accuracy metric,
  refer to the :class:`TopKAccuracy` docstring.

  Example::

    example = {'y': jnp.array([1, 2, 2, 1, 3, 0])}
    prediction = jnp.array([[0, 1, 0.5, 0], [1, 0.5, 0, 0], [0.8, 0, 0.7, 0],
                            [0.5, 1, 0, 0], [0, 0.5, 0, 1], [0.5, 0, 0.9, 0]])
    logits_mask = (0., 0., 0., jnp.NINF)
    metric = SequenceTokenTopKAccuracy(k=2, logits_mask=logits_mask)
    print(metric.evaluate_example(example, prediction))
    # MeanStat(accum=3, weight=5) => 0.6

    per_position_metric = SequenceTokenTopKAccuracy(k=2, logits_mask=logits_mask, per_position=True)
    print(per_position_metric.evaluate_example(example, prediction))
    # MeanStat(accum=[1., 0., 1., 1., 0., 0.], weight=[1., 1., 1., 1., 1., 0.]) => [1., 0., 1., 1., 0., 0.]

  Attributes:
    k: Number of top elements to look at for computing accuracy.
    target_key: Key name in ``example`` for target.
    pred_key: Key name in ``prediction`` for unnormalized model output pred.
    masked_target_values: Target values that should be ignored in computation.
      This is typically used to ignore padding values in computation.
    logits_mask: Mask of shape [num_classes] to be applied for preds. This is
      typically used to discount predictions for out-of-vocabulary tokens.
    per_position: Whether to keep output statistic per position or sum across
      positions for the entire sequence.
  """
  k: int
  target_key: str = 'y'
  pred_key: Optional[str] = None
  masked_target_values: Tuple[int, ...] = (0,)
  # logits_mask cannot be a jnp.ndarray nor np.ndarray because they are not
  # hashable and `Metric`s must be hashable for `evaluate_model`.
  logits_mask: Optional[Tuple[float, ...]] = None
  per_position: bool = False

  def zero(self) -> MeanStat:
    return MeanStat.new(0., 0.)

  def evaluate_example(self, example: SingleExample,
                       prediction: SinglePrediction) -> MeanStat:
    """Computes token top k accuracy for a single sequence example.

    Args:
      example: One example with target in range [0, num_classes) of shape
        [max_length].
      prediction: Unnormalized prediction for ``example`` of shape
        [max_length, num_classes]

    Returns:
      MeanStat for token top k accuracy for a single sequence example or at each
        token position if ``per_position`` is ``True``.
    """
    target = example[self.target_key]
    pred = prediction if self.pred_key is None else prediction[self.pred_key]
    if self.logits_mask is not None:
      logits_mask = jnp.array(self.logits_mask)
      pred += logits_mask
    target_weight = get_target_weight(target, self.masked_target_values)
    top_k_pred = jnp.argsort(-pred, axis=1)[:, :self.k]
    correct = jnp.any(
        jnp.transpose(top_k_pred) == target, axis=0).astype(jnp.float32)
    if self.per_position:
      return MeanStat.new(correct * target_weight, target_weight)
    return MeanStat.new(
        jnp.sum(correct * target_weight), jnp.sum(target_weight))


@dataclasses.dataclass
class SequenceTokenCount(Metric):
  """Metric for count of non masked tokens for a sequence example.

  Example::

    example = {'y': jnp.array([1, 2, 2, 3, 4, 0, 0])}
    prediction = jnp.array([])  # Unused.
    metric = SequenceTokenCount(masked_target_values=(0, 2))
    print(metric.evaluate_example(example, prediction))
    # SumStat(accum=3) => 3

  Attributes:
    target_key: Key name in ``example`` for target.
    masked_target_values: Target values that should be ignored in computation.
      This is typically used to ignore padding values in computation.
  """
  target_key: str = 'y'
  masked_target_values: Tuple[int, ...] = (0,)

  def zero(self) -> SumStat:
    return SumStat.new(0.)

  def evaluate_example(self, example: SingleExample,
                       prediction: SinglePrediction) -> SumStat:
    """Computes total number of non masked tokens in a single sequence example.

    Args:
      example: One example with target in range [0, num_classes) of shape [1].
      prediction: Unnormalized prediction for ``example`` of shape [num_classes]

    Returns:
      SumStat for count of non masked tokens for a single sequence example.
    """
    del prediction
    target = example[self.target_key]
    target_weight = get_target_weight(target, self.masked_target_values)
    return SumStat.new(jnp.sum(target_weight))


@dataclasses.dataclass
class SequenceCount(Metric):
  """Metric for count of non masked sequences.

  Example::

    example = {'y': jnp.array([1, 2, 2, 3, 4, 0, 0])}
    empty_example = {'y': jnp.array([0, 0, 0, 0, 0, 0, 0])}
    prediction = jnp.array([])  # Unused.
    metric = metrics.SequenceCount(masked_target_values=(0, 2))
    print(metric.evaluate_example(example, prediction))
    # SumStat(accum=1)
    print(metric.evaluate_example(empty_example, prediction))
    # SumStat(accum=0)

  Attributes:
    target_key: Key name in ``example`` for target.
    masked_target_values: Target values that should be ignored in computation.
      This is typically used to ignore padding values in computation.
  """
  target_key: str = 'y'
  masked_target_values: Tuple[int, ...] = (0,)

  def zero(self) -> SumStat:
    return SumStat.new(0.)

  def evaluate_example(self, example: SingleExample,
                       prediction: SinglePrediction) -> SumStat:
    """Counts non masked sequences.

    Args:
      example: One example with target in range [0, num_classes) of shape [1].
      prediction: Unnormalized prediction for ``example`` of shape [num_classes]

    Returns:
      SumStat for count of non masked sequence. This will be 0 or 1.
    """
    del prediction
    target = example[self.target_key]
    target_weight = get_target_weight(target, self.masked_target_values)
    return SumStat.new(jnp.any(target_weight).astype(jnp.float32))


@dataclasses.dataclass
class SequenceTruncationRate(Metric):
  """Metric for truncation rate for a sequence example.

  Example::

    example = {'y': jnp.array([1, 2, 2, 3, 3, 3, 4])}
    truncated_example = {'y': jnp.array([1, 2, 2, 3, 3, 3, 3])}
    prediction = jnp.array([])  # Unused.
    metric = SequenceTruncationRate(eos_target_value=4)
    print(metric.evaluate_example(example, prediction))
    # MeanStat(accum=0, weight=1) => 0
    print(metric.evaluate_example(truncated_example, prediction))
    # MeanStat(accum=1, weight=1) => 1

  Attributes:
    eos_target_value: Target value denoting end of sequence. Truncated sequences
      will not have this value.
    target_key: Key name in ``example`` for target.
    masked_target_values: Target values that should be ignored in computation.
      This is typically used to ignore padding values in computation.
  """
  eos_target_value: int
  target_key: str = 'y'
  masked_target_values: Tuple[int, ...] = (0,)

  def zero(self) -> MeanStat:
    return MeanStat.new(0., 0.)

  def evaluate_example(self, example: SingleExample,
                       prediction: SinglePrediction) -> MeanStat:
    """Computes truncation rate for a single sequence example.

    Args:
      example: One example with target in range [0, num_classes) of shape [1].
      prediction: Unnormalized prediction for ``example`` of shape [num_classes]

    Returns:
      MeanStat for truncation rate for a single sequence.
    """
    del prediction
    target = example[self.target_key]
    target_weight = get_target_weight(target, self.masked_target_values)
    not_empty = jnp.sum(jnp.any(target_weight))
    target_is_truncated = jnp.all(target != self.eos_target_value)
    return MeanStat.new(target_is_truncated * not_empty, not_empty)


@dataclasses.dataclass
class SequenceTokenOOVRate(Metric):
  """Metric for out-of-vocabulary (OOV) rate for a sequence example.

  Example::

    example = {'y': jnp.array([1, 2, 2, 3, 4, 0, 0])}
    prediction = jnp.array([])  # Unused.
    metric = SequenceTokenOOVRate(oov_target_values=(2,))
    print(metric.evaluate_example(example, prediction))
    # MeanStat(accum=2, weight=5) => 0.4

    per_position_metric = SequenceTokenOOVRate(oov_target_values=(2,), per_position=True)
    print(per_position_metric.evaluate_example(example, prediction))
    # MeanStat(accum=[0., 1., 1., 0., 0., 0., 0.], weight=[1., 1., 1., 1., 1., 0., 0.]) => [0. 1. 1. 0. 0. 0. 0.]

  Attributes:
    oov_target_values: Target values denoting out-of-vocabulary values.
    target_key: Key name in ``example`` for target.
    masked_target_values: Target values that should be ignored in computation.
      This is typically used to ignore padding values in computation.
    per_position: Whether to keep output statistic per position or sum across
      positions for the entire sequence.
  """
  oov_target_values: Tuple[int, ...]
  target_key: str = 'y'
  masked_target_values: Tuple[int, ...] = (0,)
  per_position: bool = False

  def zero(self) -> MeanStat:
    return MeanStat.new(0., 0.)

  def evaluate_example(self, example: SingleExample,
                       prediction: SinglePrediction) -> MeanStat:
    """Computes token out of vocabulary rate for a single sequence example.

    Args:
      example: One example with target in range [0, num_classes) of shape [1].
      prediction: Unnormalized prediction for ``example`` of shape [num_classes]

    Returns:
      MeanStat for token out of vocabulary rate for a single sequence or at each
        token position if ``per_position`` is ``True``.
    """
    del prediction
    target = example[self.target_key]
    target_weight = get_target_weight(target, self.masked_target_values)
    target_oov = jnp.ones_like(target, dtype=jnp.float32)
    for oov_value in self.oov_target_values:
      target_oov *= (target == oov_value)
    if self.per_position:
      return MeanStat.new(target_oov * target_weight, target_weight)
    return MeanStat.new(
        jnp.sum(target_oov * target_weight), jnp.sum(target_weight))


@dataclasses.dataclass
class SequenceLength(Metric):
  """Metric for length for a sequence example.

  Example::

    example = {'y': jnp.array([1, 2, 3, 4, 0, 0])}
    prediction = jnp.array([])  # Unused.
    metric = SequenceLength()
    print(metric.evaluate_example(example, prediction))
    # MeanStat(accum=4, weight=1) => 4

  Attributes:
    target_key: Key name in ``example`` for target.
    masked_target_values: Target values that should be ignored in computation.
      This is typically used to ignore padding values in computation.
  """
  target_key: str = 'y'
  masked_target_values: Tuple[int, ...] = (0,)

  def zero(self) -> MeanStat:
    return MeanStat.new(0., 0.)

  def evaluate_example(self, example: SingleExample,
                       prediction: SinglePrediction) -> MeanStat:
    """Computes non masked length for a single sequence example.

    Args:
      example: One example with target in range [0, num_classes) of shape [1].
      prediction: Unnormalized prediction for ``example`` of shape [num_classes]

    Returns:
      MeanStat for non masked length for a single sequence.
    """
    del prediction
    target = example[self.target_key]
    target_weight = get_target_weight(target, self.masked_target_values)
    return MeanStat.new(jnp.sum(target_weight), jnp.sum(jnp.any(target_weight)))


@dataclasses.dataclass
class PerDomainMetric(Metric):
  """Turns a base metric into one that groups results by domain.

  This is useful in algorithms such as AgnosticFedAvg.

  ``example`` is expected to contain a feature named
  :attr:`~PerDomainMetric.domain_id_key`, which stores the integer domain id in
  [0, num_domains). PerDomain accumulates :attr:`~PerDomainMetric.base` 's
  :class:`Stat` within each domain. If the base :class:`Metric` returns a
  :class:`Stat` whose result is of shape X, then the :class:`Stat` returned by
  PerDomain will produce a result of shape ``(num_domains,) + X``.
  See :ref:`batching-stats` for the higher rank :class:`Stat` mechanism enabling
  this.

  Example::

    per_domain_accuracy = PerDomain(metrics.Accuracy(), num_domains=3)
    batch_example = {
        'domain_id': jnp.array([0, 0, 1, 2]),
        'y': jnp.array([0, 1, 0, 1])
    }
    batch_prediction = jnp.array([[0., 1.], [2., 3.], [4., 5.], [6., 7.]])
    print(
        evaluate_batch(per_domain_accuracy, batch_example,
                       batch_prediction).result())
    # [0.5 0.  1. ]
  """
  base: Metric
  num_domains: int
  domain_id_key: str = 'domain_id'

  def zero(self) -> Stat:

    def broadcast_to(x):
      return jnp.broadcast_to(x, (self.num_domains,) + x.shape)

    return jax.tree_util.tree_map(broadcast_to, self.base.zero())

  def evaluate_example(self, example: SingleExample,
                       prediction: SinglePrediction) -> Stat:
    domain_mask = jax.nn.one_hot(
        example[self.domain_id_key], self.num_domains, dtype=jnp.bool_)

    def where(a, b):
      return apply_mask(domain_mask, jnp.expand_dims(a, 0),
                        jnp.expand_dims(b, 0))

    return jax.tree_util.tree_multimap(
        where, self.base.evaluate_example(example, prediction),
        self.base.zero())


@dataclasses.dataclass
class ConfusionMatrix(Metric):
  """Metric for making a Confusion Matrix.

  A confusion matrix is an nxn matrix often used to describe the performance
  of a classification model on a set of test data for which the true values are
  known. The model's predictions are represented through the columns, and the
  known data values through the rows. This allows one to view in which areas the
  model is doing well, as well as where there is room for improvement. For each
  row in the confusion matrix, if there are a lot of numbers outside of the main
  diagonal, the model is not doing so well in respect to when it is supposed to
  output that row's relative output class.

  Theoretical Example::

                Predicted P     Predicted N

    Actual P       TP               FN

    Actual N       FP               TN

    **This is for a binary classification model but the same concept applies
    to any model with n outputs. Notice that the TPs and TNs will always
    lie in the main diagonal of the matrix.

  Example::

    example = {'y': jnp.array(2)}
    prediction = jnp.array([0., 1., 0.])
    metric = ConfusionMatrix(num_classes=3)
    print(metric.evaluate_example(example, prediction))
    # SumStat(accum=DeviceArray([[0., 0., 0.],
    #                            [0., 0., 0.],
    #                            [0., 1., 0.]], dtype=float32)) => [[0. 0. 0.]
    #                                                               [0. 0. 0.]
    #                                                               [0. 1. 0.]]

  Attributes:
    target_key: Key name in ``example`` for target.
    pred_key: Key name in ``prediction`` for unnormalized model output pred.
    num_classes: Number of output classes of the model. Used to generate a
      matrix of shape [num_classes, num_classes].
  """
  num_classes: int
  target_key: str = 'y'
  pred_key: Optional[str] = None

  def zero(self) -> SumStat:
    return SumStat.new(jnp.zeros((self.num_classes, self.num_classes)))

  def evaluate_example(self, example: SingleExample,
                       prediction: SinglePrediction) -> SumStat:
    """Computes a Confusion Matrix for a single example.

    Args:
      example: One example with target in range [0, num_classes) of shape [1].
      prediction: Unnormalized prediction for ``example`` of shape [num_classes]

    Returns:
      SumStat for the Confusion Matrix for a single example.

    Raises:
      ValueError: If the num_classes attribute is not equal to the number of
        output classes of the model.
    """
    target = example[self.target_key]
    pred = prediction if self.pred_key is None else prediction[self.pred_key]
    if self.num_classes != len(pred):
      raise ValueError('Make sure num_classes is equal to the number of output '
                       f'classes of the model. num_classes: {self.num_classes} '
                       f'number of output classes of the model: {len(pred)}')
    pred_idx = jnp.argmax(pred, axis=-1)
    confusion_matrix = jnp.zeros((self.num_classes, self.num_classes))
    return SumStat.new(confusion_matrix.at[target, pred_idx].set(1))
