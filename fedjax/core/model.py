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
"""Lightweight convenience container for various model implementations."""

import collections
import functools
from typing import Any, Callable, Dict, Optional, Mapping, NamedTuple, Tuple

import dataclasses
from fedjax.core import metrics
from fedjax.core.typing import Batch
from fedjax.core.typing import Params
from fedjax.core.typing import PRNGKey
from fedjax.core.typing import Updates
import frozendict
import haiku as hk
import jax
import jax.numpy as jnp

MetricsFn = Callable[[Batch, jnp.ndarray], metrics.Metric]


class BackwardPassOutput(NamedTuple):
  """Struct for backward pass output that can be passed to JAX transformations.

  Attributes:
    grads: Gradients of same structure as given params.
    loss: Scalar loss for given batch.
    num_examples: Number of examples seen in a given batch.
  """
  grads: Updates
  loss: jnp.ndarray
  num_examples: float


@dataclasses.dataclass(frozen=True)
class Model:
  """Cotainer class for models.

  Works for Haiku (go/dm-haiku) and jax.experimental.stax.
  NOTE: Users should not create this directly but use `create_model_from_haiku`
  or `create_model_from_stax` instead.

  Attributes:
    init_fn: Model parameter initialization function.
    apply_fn: Function that produces model predictions for input batch.
    loss_fn: Loss function that takes input batch and model predictions and
      returns scalar loss.
    reg_fn: Regularization function that takes parameters in and returns a
      scalar regularizer value.
    metrics_fn_map: Ordered mapping of metric names to metric functions that
      take input batch and model predictions and return metric values.
    train_kwargs: Keyword arguments passed to apply for training.
    test_kwargs: Keyword arguments passed to apply for testing.
    modify_grads_fn: Callable that modifies input gradients.
  """
  init_fn: Callable[[PRNGKey], Params]
  apply_fn: Callable[..., jnp.ndarray]
  loss_fn: MetricsFn
  reg_fn: Callable[[Params], jnp.ndarray] = lambda p: 0.
  metrics_fn_map: Mapping[str, MetricsFn] = frozendict.frozendict()
  train_kwargs: Mapping[str, Any] = frozendict.frozendict()
  test_kwargs: Mapping[str, Any] = frozendict.frozendict()
  modify_grads_fn: Callable[[Updates], Updates] = lambda g: g

  def init_params(self, rng: PRNGKey) -> Params:
    return self.init_fn(rng)

  @functools.partial(jax.jit, static_argnums=0)
  def backward_pass(self, params: Params, batch: Batch,
                    rng: Optional[PRNGKey]) -> BackwardPassOutput:
    """Runs backward pass and returns BackwardPassOutput."""

    def loss_fn(p):
      preds = self.apply_fn(p, rng, batch, **self.train_kwargs)
      loss_metric = self.loss_fn(batch, preds)
      loss = loss_metric.result() + self.reg_fn(p)
      return loss, loss_metric.count

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, num_examples), grads = grad_fn(params)
    grads = self.modify_grads_fn(grads)
    return BackwardPassOutput(grads=grads, loss=loss, num_examples=num_examples)

  @functools.partial(jax.jit, static_argnums=0)
  def evaluate(self, params: Params, batch: Batch) -> Dict[str, metrics.Metric]:
    """Evaluates model on input batch."""
    rng = None
    preds = self.apply_fn(params, rng, batch, **self.test_kwargs)
    loss_metric = self.loss_fn(batch, preds)
    num_examples = loss_metric.count
    metrics_dict = collections.OrderedDict(
        loss=loss_metric,
        regularizer=metrics.MeanMetric(
            total=self.reg_fn(params), count=num_examples),
        num_examples=metrics.CountMetric(count=num_examples))
    for metric_name, metric_fn in self.metrics_fn_map.items():
      metrics_dict[metric_name] = metric_fn(batch, preds)
    return metrics_dict


def _get_defaults(reg_fn, metrics_fn_map, train_kwargs, test_kwargs):
  """Builds default values."""
  if reg_fn is None:
    reg_fn = lambda p: 0.
  metrics_fn_map = metrics_fn_map or collections.OrderedDict()
  train_kwargs = train_kwargs or {}
  test_kwargs = test_kwargs or {}
  frozen_metrics_fn_map = frozendict.frozendict(metrics_fn_map)
  frozen_train_kwargs = frozendict.frozendict(train_kwargs)
  frozen_test_kwargs = frozendict.frozendict(test_kwargs)
  return reg_fn, frozen_metrics_fn_map, frozen_train_kwargs, frozen_test_kwargs


def create_model_from_haiku(
    transformed_forward_pass: hk.Transformed,
    sample_batch: Batch,
    loss_fn: MetricsFn,
    reg_fn: Optional[Callable[[Params], jnp.ndarray]] = None,
    metrics_fn_map: Optional[Mapping[str, MetricsFn]] = None,
    train_kwargs: Optional[Mapping[str, Any]] = None,
    test_kwargs: Optional[Mapping[str, Any]] = None,
    non_trainable_module_names: Tuple[str] = ()
) -> Model:
  """Creates Model after applying defaults and haiku specific preprocessing.

  Args:
    transformed_forward_pass: Transformed forward pass from `hk.transform`.
    sample_batch: Example input batch used to determine model parameter shapes.
    loss_fn: Loss function that takes input batch and model predictions and
      returns scalar loss.
    reg_fn: Regularization function that takes parameters in and returns a
      scalar regularizer value. Defaults to no regularization.
    metrics_fn_map: Ordered mapping of metric names to metric functions that
      take input batch and model predictions and return metric values that will
      be freezed for immutability. Defaults to empty frozen dictionary.
    train_kwargs: Keyword arguments passed to model for training that will be
      freezed for immutability. Defaults to empty frozen dictionary.
    test_kwargs: Keyword arguments passed to model for testing that will be
      freezed for immutability. Defaults to empty frozen dictionary.
    non_trainable_module_names: List of `hk.Module` names whose parameters
      should not to be updated during training.

  Returns:
    Model
  """
  reg_fn, metrics_fn_map, train_kwargs, test_kwargs = _get_defaults(
      reg_fn, metrics_fn_map, train_kwargs, test_kwargs)

  def ignore_grads(grads):
    predicate = lambda module_name, *_: module_name in non_trainable_module_names
    non_trainable, trainable = hk.data_structures.partition(predicate, grads)
    non_trainable = jax.tree_map(jnp.zeros_like, non_trainable)
    return hk.data_structures.merge(non_trainable, trainable)

  def init(rng):
    return transformed_forward_pass.init(rng, sample_batch)

  return Model(init, transformed_forward_pass.apply, loss_fn, reg_fn,
               metrics_fn_map, train_kwargs, test_kwargs, ignore_grads)


def create_model_from_stax(
    stax_init_fn: Callable[..., Params],
    stax_apply_fn: Callable[..., jnp.ndarray],
    sample_shape: Tuple[int, ...],
    loss_fn: MetricsFn,
    input_key: str = 'x',
    reg_fn: Optional[Callable[[Params], jnp.ndarray]] = None,
    metrics_fn_map: Optional[Mapping[str, MetricsFn]] = None,
    train_kwargs: Optional[Mapping[str, Any]] = None,
    test_kwargs: Optional[Mapping[str, Any]] = None) -> Model:
  """Creates Model after applying defaults and stax specific preprocessing.

  Args:
    stax_init_fn: Initialization function returned from `stax.serial`.
    stax_apply_fn: Model forward_pass pass function returned from `stax.serial`.
    sample_shape: The expected shape of the input to the model.
    loss_fn: Loss function that takes input batch and model predictions and
      returns scalar loss.
    input_key: Key name for the input from batch mapping.
    reg_fn: Regularization function that takes parameters in and returns a
      scalar regularizer value. Defaults to no regularization.
    metrics_fn_map: Ordered mapping of metric names to metric functions that
      take input batch and model predictions and return metric values that will
      be freezed for immutability. Defaults to empty frozen dictionary.
    train_kwargs: Keyword arguments passed to model for training that will be
      freezed for immutability. Defaults to empty frozen dictionary.
    test_kwargs: Keyword arguments passed to model for testing that will be
      freezed for immutability. Defaults to empty frozen dictionary.

  Returns:
    Model
  """
  reg_fn, metrics_fn_map, train_kwargs, test_kwargs = _get_defaults(
      reg_fn, metrics_fn_map, train_kwargs, test_kwargs)

  def init(rng):
    _, params = stax_init_fn(rng, sample_shape)
    return params

  def apply(params, rng, batch, **kwargs):
    return stax_apply_fn(params, batch[input_key], rng=rng, **kwargs)

  return Model(init, apply, loss_fn, reg_fn, metrics_fn_map, train_kwargs,
               test_kwargs)
