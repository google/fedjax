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
"""Lightweight convenience container for various model implementations."""

import collections
import functools
from typing import Any, Callable, Dict, Iterable, Optional, Mapping, Tuple

from fedjax import core

import haiku as hk
import immutabledict
import jax
import jax.numpy as jnp

# Typically unnormalized model forward pass output.
TrainOutput = jnp.ndarray
EvalOutput = jnp.ndarray


@core.dataclass
class Model:
  """Container class for models.

  `Model` exists to provide easy access to predefined neural network models.
  It is meant to contain all the information needed for standard centralized
  training and evaluation. Non-standard training methods can be built upon the
  information avaiable in `Model` along with any additional information
  (e.g. interpolation can be implemented as a composition of two models along
  with an interpolation weight).

  Works for Haiku (go/dm-haiku) and jax.experimental.stax.

  The expected usage of `Model` is as follows:

  ```
  # Training.
  step_size = 0.1
  rng = jax.random.PRNGKey(0)
  params = model.init_params(rng)

  def loss(params, batch, rng):
    preds = model.apply_for_train(params, batch, rng)
    return jnp.sum(model.train_loss(batch, preds))

  grad_fn = jax.grad(loss)
  for batch in batches:
    rng, use_rng = jax.random.split(rng)
    grads = grad_fn(params, batch, use_rng)
    params = jax.tree_util.tree_multimap(lambda a, b: a - step_size * b,
                                         params, grads)

  # Evaluation.
  for metric_name, metric_fn in model.eval_metrics.items():
    metric = metric_fn(batch, model.apply_for_eval(params, batch))
    print(metric_name, metric.result())
  ```

  The following is an example using `Model` compositionally as a building block
  to impelement model interpolation.

  ```
  def interpolate(model_1, model_2, init_weight):

    @jax.jit
    def init_params(rng):
      rng_1, rng_2 = jax.random.split(rng)
      params_1 = model_1.init_params(rng_1)
      params_2 = model_2.init_params(rng_2)
      return params_1, params_2, init_weight

    @jax.jit
    def apply_for_train(params, input, rng):
      rng_1, rng_2 = jax.random.split(rng)
      params_1, params_2, weight = params
      return (model_1.apply_for_train(params_1, input, rng_1) * weight +
              model_2.apply_for_train(params_1, input, rng_2) * (1 - weight))

    @jax.jit
    def apply_for_eval(params, input):
      params_1, params_2, weight = params
      return (model_1.apply_for_eval(params_1, input) * weight +
              model_2.apply_for_eval(params_2, input) * (1 - weight))

    return Model(init_params, apply_for_train, apply_for_eval,
                 model_1.train_loss, model_1.eval_metrics)

  model = interpolate(model_1, model_2, init_weight=0.5)
  ```

  Attributes:
    init_params: Initialization function that takes a seed `PRNGKey` and returns
      a PyTree of initialized parameters (i.e. model weights). These parameters
      will be passed as input into `apply_for_train` and `apply_for_eval`. Any
      trainable weights for a model that are modified in the training loop
      should be contained inside of these parameters.
    apply_for_train: Function that takes the parameters PyTree, batch of
      examples, and `PRNGKey` as inputs and outputs the model predictions for
      training that are then passed into `train_loss`. This considers strategies
      such as dropout.
    apply_for_eval: Function that usually takes the parameters PyTree and batch
      of examples as inputs and outputs the model predictions for evaluation
      that are then passed into the metric functions in `eval_metric_fns`. This
      is defined separately from `apply_for_train` to avoid having to specify
      inputs like `PRNGKey` that are not used in evaluation.
    train_loss: Loss function for training that takes batch of examples and
      model output from `apply_for_train` as input that outputs per example
      loss. This will typically called inside a `jax.grad` wrapped function to
      compute gradients.
    eval_metric_fns: Ordered mapping of evaluation metric names to functions
      that take a batch of examples and model output from `apply_for_eval` as
      input and output `Metric`.
  """
  init_params: Callable[[core.PRNGKey], core.Params]
  apply_for_train: Callable[[core.Params, core.Batch, Optional[core.PRNGKey]],
                            TrainOutput]
  apply_for_eval: Callable[[core.Params, core.Batch], EvalOutput]
  train_loss: Callable[[core.Batch, TrainOutput], jnp.ndarray]
  eval_metrics: Mapping[str, Callable[[core.Batch, EvalOutput], core.Metric]]


def create_model_from_haiku(
    transformed_forward_pass: hk.Transformed,
    sample_batch: core.Batch,
    train_loss: Callable[[core.Batch, TrainOutput], jnp.ndarray],
    eval_metric_fns: Optional[Mapping[str, Callable[[core.Batch, EvalOutput],
                                                    core.Metric]]] = None,
    train_kwargs: Optional[Mapping[str, Any]] = None,
    eval_kwargs: Optional[Mapping[str, Any]] = None) -> Model:
  """Creates Model after applying defaults and haiku specific preprocessing.

  Args:
    transformed_forward_pass: Transformed forward pass from `hk.transform`.
    sample_batch: Example input used to determine model parameter shapes.
    train_loss: Loss function for training that outputs per example loss.
    eval_metric_fns: Ordered mapping of evaluation metric names to functions
      that take a batch of examples and model output from `apply_for_eval` as
      input and output `Metric`. This mapping will be made immutable. Defaults
      to empty immutable dictionary.
    train_kwargs: Keyword arguments passed to model for training.
    eval_kwargs: Keyword arguments passed to model for evaluation.

  Returns:
    Model
  """
  eval_metric_fns = eval_metric_fns or {}
  train_kwargs = train_kwargs or {}
  eval_kwargs = eval_kwargs or {}

  @jax.jit
  def init_params(rng):
    return transformed_forward_pass.init(rng, sample_batch)

  @jax.jit
  def apply_for_train(params, batch, rng=None):
    return transformed_forward_pass.apply(params, rng, batch, **train_kwargs)

  @jax.jit
  def apply_for_eval(params, batch):
    return transformed_forward_pass.apply(params, None, batch, **eval_kwargs)

  return Model(init_params, apply_for_train, apply_for_eval, train_loss,
               immutabledict.immutabledict(eval_metric_fns))


def create_model_from_stax(stax_init: Callable[..., core.Params],
                           stax_apply: Callable[..., jnp.ndarray],
                           sample_shape: Tuple[int, ...],
                           train_loss: Callable[[core.Batch, TrainOutput],
                                                jnp.ndarray],
                           eval_metric_fns: Optional[Mapping[str, Callable[
                               [core.Batch, EvalOutput], core.Metric]]] = None,
                           train_kwargs: Optional[Mapping[str, Any]] = None,
                           eval_kwargs: Optional[Mapping[str, Any]] = None,
                           input_key: str = 'x') -> Model:
  """Creates Model after applying defaults and stax specific preprocessing.

  Args:
    stax_init: Initialization function returned from `stax.serial`.
    stax_apply: Model forward_pass pass function returned from `stax.serial`.
    sample_shape: The expected shape of the input to the model.
    train_loss: Loss function for training that outputs per example loss.
    eval_metric_fns: Ordered mapping of evaluation metric names to functions
      that take a batch of examples and model output from `apply_for_eval` as
      input and output `Metric`. This mapping will be made immutable. Defaults
      to empty immutable dictionary.
    train_kwargs: Keyword arguments passed to model for training.
    eval_kwargs: Keyword arguments passed to model for evaluation.
    input_key: Key name for the input in batch mapping.

  Returns:
    Model
  """
  eval_metric_fns = eval_metric_fns or {}
  train_kwargs = train_kwargs or {}
  eval_kwargs = eval_kwargs or {}

  @jax.jit
  def init_params(rng):
    _, params = stax_init(rng, sample_shape)
    return params

  @jax.jit
  def apply_for_train(params, batch, rng=None):
    return stax_apply(params, batch[input_key], rng=rng, **train_kwargs)

  @jax.jit
  def apply_for_eval(params, batch):
    return stax_apply(params, batch[input_key], **eval_kwargs)

  return Model(init_params, apply_for_train, apply_for_eval, train_loss,
               immutabledict.immutabledict(eval_metric_fns))


@functools.partial(jax.jit, static_argnums=0)
def _evaluate_model_on_batch(model: Model, params: core.Params,
                             batch: core.Batch) -> Dict[str, core.Metric]:
  eval_output = model.apply_for_eval(params, batch)
  return collections.OrderedDict(
      (metric_name, metric_fn(batch, eval_output))
      for metric_name, metric_fn in model.eval_metrics.items())


def evaluate_model(model: Model, params: core.Params,
                   batches: Iterable[core.Batch]) -> Dict[str, jnp.ndarray]:
  """Evaluates model on multiple batches.

  Args:
    model: Model container expected to be static and unchanging.
    params: Model parameters to be evaluated.
    batches: Multiple batches to compute and aggregate evaluation metrics over.

  Returns:
    Dictionary of metric names to jax numpy array values.
  """
  metrics = None
  for batch in batches:
    batch_metrics = _evaluate_model_on_batch(model, params, batch)
    if metrics is None:
      metrics = batch_metrics
    else:
      metrics = collections.OrderedDict(
          (k, metrics[k].merge(batch_metrics[k])) for k in batch_metrics)
  return collections.OrderedDict((k, metrics[k].result()) for k in metrics)
