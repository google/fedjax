# Copyright 2020 Google LLC
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
"""Metrics."""

import abc
import numbers
from typing import Any, Optional, Tuple

import dataclasses
import jax
import jax.numpy as jnp

# Small constant to add to denominator to avoid division by 0.
_SAFE_DIVIDE = 1e-10


# Forked and slimmed down from
# https://flax.readthedocs.io/en/latest/_modules/flax/struct.html#dataclass
# https://github.com/google/jax/issues/2371
def dataclass(clz: type):
  """Creates a dataclass which can be passed to functional transformations."""
  data_clz = dataclasses.dataclass(frozen=True)(clz)
  meta_fields = []
  data_fields = []
  for name, field_info in data_clz.__dataclass_fields__.items():
    is_pytree_node = field_info.metadata.get('pytree_node', True)
    if is_pytree_node:
      data_fields.append(name)
    else:
      meta_fields.append(name)

  def replace(self, **updates):
    """"Returns a new object replacing the specified fields with new values."""
    return dataclasses.replace(self, **updates)

  data_clz.replace = replace

  def iterate_clz(x):
    meta = tuple(getattr(x, name) for name in meta_fields)
    data = tuple(getattr(x, name) for name in data_fields)
    return data, meta

  def clz_from_iterable(meta, data):
    meta_args = tuple(zip(meta_fields, meta))
    data_args = tuple(zip(data_fields, data))
    kwargs = dict(meta_args + data_args)
    return data_clz(**kwargs)

  jax.tree_util.register_pytree_node(data_clz, iterate_clz, clz_from_iterable)
  return data_clz


class Metric(metaclass=abc.ABCMeta):
  """Interface for all metric containers (e.g.

  accuracy).

  `Metric` stores intermediate values as well as methods for accumulation and
  final result computation.
  """

  @abc.abstractmethod
  def merge(self, other: 'Metric') -> 'Metric':
    """Merges `self` and `other` into a new single accumulated metric."""

  @abc.abstractmethod
  def result(self) -> jnp.ndarray:
    """Computes final metric result from intermediate values."""

  def __str__(self) -> str:
    """Returns human readable string representation of metric."""
    return f'{repr(self)} => {self.result()}'


def _is_scalar(x):
  if isinstance(x, jnp.ndarray):
    return x.ndim == 0
  return isinstance(x, numbers.Number)


@dataclass
class MeanMetric(Metric):
  """Implementation for metrics that are reduced by averaging (total / count).

  Attributes:
    total: Scalar sum of intermediate values.
    count: Scalar number of intermediate values.
  """

  total: jnp.ndarray
  count: jnp.ndarray

  def __post_init__(self):
    if not (_is_scalar(self.total) and _is_scalar(self.count)):
      raise TypeError('total and count must both be scalars.')

  @classmethod
  def from_values(cls,
                  values: jnp.ndarray,
                  weights: Optional[jnp.ndarray] = None) -> 'MeanMetric':
    """Constructs MeanMetric from intermediate values and optional weights.

    Args:
      values: Array of intermediate values.
      weights: Array of weights for each intermediate value of the same shape as
        values. Defaults to unweighted.

    Returns:
      MeanMetric for (possibly weighted) average of values.
    """
    if weights is None:
      weights = jnp.ones_like(values)
    return cls(total=jnp.sum(values * weights), count=jnp.sum(weights))

  def merge(self, other: 'MeanMetric') -> 'MeanMetric':
    return type(self)(
        total=self.total + other.total, count=self.count + other.count)

  def result(self) -> jnp.ndarray:
    return self.total / jnp.maximum(self.count, _SAFE_DIVIDE)


@dataclass
class CountMetric(Metric):
  """Implementation for counter metrics (e.g. num_out_of_vocabulary_words)."""

  count: jnp.ndarray

  def __post_init__(self):
    if not _is_scalar(self.count):
      raise TypeError('count must be a scalar.')

  def merge(self, other: 'CountMetric') -> 'CountMetric':
    return type(self)(count=self.count + other.count)

  def result(self) -> jnp.ndarray:
    return self.count


def _unreduced_cross_entropy_loss_fn(targets: jnp.ndarray,
                                     preds: jnp.ndarray) -> jnp.ndarray:
  """Returns unreduced cross entropy loss."""
  num_classes = preds.shape[-1]
  log_preds = jax.nn.log_softmax(preds)
  one_hot_targets = jax.nn.one_hot(targets, num_classes)
  return -jnp.sum(one_hot_targets * log_preds, axis=-1)


def cross_entropy_loss_fn(targets: jnp.ndarray,
                          preds: jnp.ndarray) -> MeanMetric:
  """Computes cross entropy loss.

  Args:
    targets: Target values of shape [batch_size, ...] in range [0, num_classes).
    preds: Unnormalized model output of shape [batch_size, ..., num_classes].

  Returns:
    Metric for loss.
  """
  unreduced_loss = _unreduced_cross_entropy_loss_fn(targets, preds)
  return MeanMetric.from_values(unreduced_loss)


def masked_cross_entropy_loss_fn(
    targets: jnp.ndarray, preds: jnp.ndarray,
    mask_values: Tuple[int, ...] = ()) -> MeanMetric:
  """Computes cross entropy loss after discounting masked values.

  Args:
    targets: Target values of shape [batch_size, ...] in range [0, num_classes).
    preds: Unnormalized model output of shape [batch_size, ..., num_classes].
    mask_values: Target values to be masked and not counted in loss.

  Returns:
    Metric for masked loss.
  """
  weights = jnp.ones_like(targets, dtype=preds.dtype)
  for mv in mask_values:
    weights *= (targets != mv)
  unreduced_loss = _unreduced_cross_entropy_loss_fn(targets, preds)
  return MeanMetric.from_values(unreduced_loss, weights=weights)


def accuracy_fn(targets: jnp.ndarray, preds: jnp.ndarray) -> MeanMetric:
  """Computes accuracy.

  Args:
    targets: Target values of shape [batch_size, ...] in range [0, num_classes).
    preds: Unnormalized model output of shape [batch_size, ..., num_classes].

  Returns:
    Metric for accuracy.
  """
  pred_class = jnp.argmax(preds, axis=-1)
  return MeanMetric.from_values(pred_class == targets)


def masked_accuracy_fn(
    targets: jnp.ndarray,
    preds: jnp.ndarray,
    mask_values: Tuple[int, ...] = (),
) -> MeanMetric:
  """Computes accuracy after discounting masked values.

  Args:
    targets: Target values of shape [batch_size, ...] in range [0, num_classes).
    preds: Unnormalized model output of shape [batch_size, ..., num_classes].
    mask_values: Target values to be masked and not counted in accuracy.

  Returns:
    Metric for masked accuracy.
  """
  weights = jnp.ones_like(targets, dtype=preds.dtype)
  for mv in mask_values:
    weights *= (targets != mv)
  pred_class = jnp.argmax(preds, axis=-1)
  return MeanMetric.from_values(pred_class == targets, weights=weights)


def masked_accuracy_fn_with_logits_mask(
    targets: jnp.ndarray,
    preds: jnp.ndarray,
    logits_mask: jnp.ndarray,
    mask_values: Tuple[int, ...] = (),
) -> MeanMetric:
  """Computes accuracy after discounting masked values.

  Args:
    targets: Target values of shape [batch_size, ...] in range [0, num_classes).
    preds: Unnormalized model output of shape [batch_size, ..., num_classes].
    logits_mask: Mask of shape [num_classes] to be applied for preds.
    mask_values: Target values to be masked and not counted in accuracy.

  Returns:
    Metric for masked accuracy with logits mask.
  """
  weights = jnp.ones_like(targets, dtype=preds.dtype)
  for mv in mask_values:
    weights *= (targets != mv)
  preds = preds + logits_mask
  pred_class = jnp.argmax(preds, axis=-1)
  return MeanMetric.from_values(pred_class == targets, weights=weights)


def masked_count(
    targets: jnp.ndarray, mask_values: Tuple[Any, ...] = ()) -> CountMetric:
  """Counts total number of non masked targets."""
  weights = jnp.ones_like(targets, dtype=jnp.int32)
  for mv in mask_values:
    weights *= (targets != mv)
  return CountMetric(count=jnp.sum(weights))


def truncation_rate(targets: jnp.ndarray, eos_value: int,
                    pad_value: int) -> MeanMetric:
  """Computes the proportion of sequence examples that were truncated.

  Args:
    targets: Target values of shape [batch_size, sequence_length, ...].
    eos_value: Target value denoting end of sequence. Truncated sequences will
      not have this value.
    pad_value: Optional target value for padding to discount empty sequences.

  Returns:
    Metric for trucation rate.
  """
  not_empty = jnp.any(targets != pad_value, axis=1)
  is_truncated = jnp.all(targets != eos_value, axis=1) * not_empty
  return MeanMetric(total=jnp.sum(is_truncated), count=jnp.sum(not_empty))


def oov_rate(
    targets: jnp.ndarray,
    oov_values: Tuple[int, ...],
    mask_values: Tuple[int, ...] = ()) -> MeanMetric:
  """Computes proportion of non masked tokens that are out of vocabulary.

  Args:
    targets: Target values of shape [batch_size, sequence_length, ...].
    oov_values: Target values denoting out of vocabulary values.
    mask_values: Target values to be masked and not counted in metric.

  Returns:
    Metric for out of vocabulary rate.
  """
  weights = jnp.ones_like(targets, dtype=jnp.float32)
  for mv in mask_values:
    weights *= (targets != mv)
  num_non_masked = jnp.sum(weights)
  for ov in oov_values:
    weights *= (targets == ov)
  num_oov = jnp.sum(weights)
  return MeanMetric(total=num_oov, count=num_non_masked)


def sequence_length(targets: jnp.ndarray, pad_value: int) -> MeanMetric:
  """Computes length of sequence examples by number of non-pad tokens."""
  non_pad_mask = targets != pad_value
  not_empty = jnp.any(non_pad_mask, axis=1)
  num_non_pad = jnp.sum(non_pad_mask, axis=1)
  return MeanMetric(total=jnp.sum(num_non_pad), count=jnp.sum(not_empty))
