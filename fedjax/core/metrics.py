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

from typing import Tuple

from fedjax.core.typing import Batch
import jax
import jax.numpy as jnp

# Small constant to add to denominator to avoid division by 0.
_SAFE_DIVIDE = 1e-10


def _unreduced_cross_entropy_loss_fn(targets: jnp.ndarray,
                                     preds: jnp.ndarray) -> jnp.ndarray:
  """Returns unreduced cross entropy loss."""
  num_classes = preds.shape[-1]
  log_preds = jax.nn.log_softmax(preds)
  one_hot_targets = jax.nn.one_hot(targets, num_classes)
  return -jnp.sum(one_hot_targets * log_preds, axis=-1)


def cross_entropy_loss_fn(targets: jnp.ndarray,
                          preds: jnp.ndarray) -> jnp.ndarray:
  """Computes cross entropy loss.

  Args:
    targets: Target values of shape [batch_size, ...] in range [0, num_classes).
    preds: Unnormalized model output of shape [batch_size, ..., num_classes].

  Returns:
    Scalar loss.
  """
  unreduced_loss = _unreduced_cross_entropy_loss_fn(targets, preds)
  return jnp.mean(unreduced_loss)


def masked_cross_entropy_loss_fn(targets: jnp.ndarray,
                                 preds: jnp.ndarray,
                                 mask_values: Tuple[int, ...] = (),
                                 reduce: bool = True) -> jnp.ndarray:
  """Computes cross entropy loss after discounting masked values.

  Args:
    targets: Target values of shape [batch_size, ...] in range [0, num_classes).
    preds: Unnormalized model output of shape [batch_size, ..., num_classes].
    mask_values: Target values to be masked and not counted in loss.
    reduce: Whether to reduce output loss to a scalar or not.

  Returns:
    If reduce is True, loss of shape []. Else, loss of shape targets.shape.
  """
  weights = jnp.ones_like(targets, dtype=preds.dtype)
  for mv in mask_values:
    weights *= (targets != mv)
  unreduced_loss = _unreduced_cross_entropy_loss_fn(targets, preds)
  scaled_unreduced_loss = unreduced_loss * weights
  if not reduce:
    return scaled_unreduced_loss
  return jnp.sum(scaled_unreduced_loss) / (jnp.sum(weights) + _SAFE_DIVIDE)


def accuracy_fn(targets: jnp.ndarray, preds: jnp.ndarray) -> jnp.ndarray:
  """Computes accuracy.

  Args:
    targets: Target values of shape [batch_size, ...] in range [0, num_classes).
    preds: Unnormalized model output of shape [batch_size, ..., num_classes].

  Returns:
    Scalar accuracy.
  """
  pred_class = jnp.argmax(preds, axis=-1)
  return jnp.mean(pred_class == targets)


def masked_accuracy_fn(
    targets: jnp.ndarray,
    preds: jnp.ndarray,
    mask_values: Tuple[int, ...] = (),
) -> jnp.ndarray:
  """Computes accuracy after discounting masked values.

  Args:
    targets: Target values of shape [batch_size, ...] in range [0, num_classes).
    preds: Unnormalized model output of shape [batch_size, ..., num_classes].
    mask_values: Target values to be masked and not counted in accuracy.

  Returns:
    Scaled scalar accuracy.
  """
  weights = jnp.ones_like(targets, dtype=preds.dtype)
  for mv in mask_values:
    weights *= (targets != mv)
  pred_class = jnp.argmax(preds, axis=-1)
  correct_class = pred_class == targets
  return jnp.sum(correct_class * weights) / (jnp.sum(weights) + _SAFE_DIVIDE)


def masked_accuracy_fn_with_logits_mask(
    targets: jnp.ndarray,
    preds: jnp.ndarray,
    logits_mask: jnp.ndarray,
    mask_values: Tuple[int, ...] = (),
) -> jnp.ndarray:
  """Computes accuracy after discounting masked values.

  Args:
    targets: Target values of shape [batch_size, ...] in range [0, num_classes).
    preds: Unnormalized model output of shape [batch_size, ..., num_classes].
    logits_mask: Mask of shape [num_classes] to be applied for preds.
    mask_values: Target values to be masked and not counted in accuracy.

  Returns:
    Scaled scalar accuracy.
  """
  weights = jnp.ones_like(targets, dtype=preds.dtype)
  for mv in mask_values:
    weights *= (targets != mv)
  preds = preds + logits_mask
  pred_class = jnp.argmax(preds, axis=-1)
  correct_class = pred_class == targets
  return jnp.sum(correct_class * weights) / (jnp.sum(weights) + _SAFE_DIVIDE)


def get_target_label_from_batch(func):
  """Wraps input function to get target jnp.ndarray from batch."""

  def wrapper(batch: Batch, *args, **kwargs):
    return func(batch['y'], *args, **kwargs)

  return wrapper


def masked_weight_fn(
    batch: Batch, mask_values: Tuple[int, ...] = ()) -> jnp.ndarray:
  """Computes weight of a batch discounting masked values.

  Args:
    batch: Input batch.
    mask_values: Target values to be masked and not counted in accuracy.

  Returns:
    Masked weight of the batch.
  """
  weights = jnp.ones_like(batch['y'], dtype=float)
  for mv in mask_values:
    weights *= (batch['y'] != mv)
  return jnp.sum(weights)
