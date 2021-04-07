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

import functools
from typing import Any, Callable, Dict, Iterable, Optional, Mapping, Tuple

from fedjax import core
from fedjax.experimental import metrics
from fedjax.experimental.typing import BatchExample
from fedjax.experimental.typing import BatchPrediction

import haiku as hk
import immutabledict
import jax
import jax.numpy as jnp

Params = core.Params
PRNGKey = core.PRNGKey

# Typically unnormalized model forward pass output.
BatchTrainOutput = jnp.ndarray
BatchEvalPrediction = BatchPrediction


@core.dataclass
class Model:
  """Container class for models.

  `Model` exists to provide easy access to predefined neural network models.
  It is meant to contain all the information needed for standard centralized
  training and evaluation. Non-standard training methods can be built upon the
  information avaiable in `Model` along with any additional information
  (e.g. interpolation can be implemented as a composition of two models along
  with an interpolation weight).

  Works for Haiku (go/dm-haiku) and jax.experimental.stax. We strongly recommend
  using the `Model.new()` factory method to construct `Model` objects.

  The expected usage of `Model` is as follows:

  ```
  # Training.
  step_size = 0.1
  rng = jax.random.PRNGKey(0)
  params = model.init(rng)

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
  print(fedjax.experimental.model.evaluate_model(model, params, batches))
  # Example output:
  # {'loss': 2.3, 'accuracy': 0.2}
  ```

  The following is an example using `Model` compositionally as a building block
  to impelement model interpolation.

  ```
  def interpolate(model_1, model_2, init_weight):

    @jax.jit
    def init(rng):
      rng_1, rng_2 = jax.random.split(rng)
      params_1 = model_1.init(rng_1)
      params_2 = model_2.init(rng_2)
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

    return fedjax.experimental.model.Model.new(init,
                                               apply_for_train,
                                               apply_for_eval,
                                               model_1.train_loss,
                                               model_1.eval_metrics)

  model = interpolate(model_1, model_2, init_weight=0.5)
  ```

  Attributes:
    init: Initialization function that takes a seed `PRNGKey` and returns a
      PyTree of initialized parameters (i.e. model weights). These parameters
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
    eval_metrics: Ordered mapping of evaluation metric names to `Metric`s. These
      `Metric`s are defined for single examples and will be consumed in
      `evaluate_model`.
  """
  init: Callable[[PRNGKey], Params]
  apply_for_train: Callable[[Params, BatchExample, Optional[PRNGKey]],
                            BatchTrainOutput]
  apply_for_eval: Callable[[Params, BatchExample], BatchEvalPrediction]
  train_loss: Callable[[BatchExample, BatchTrainOutput], jnp.ndarray]
  eval_metrics: Mapping[str, metrics.Metric]

  @classmethod
  def new(cls, init: Callable[[PRNGKey], Params],
          apply_for_train: Callable[[Params, BatchExample, Optional[PRNGKey]],
                                    BatchTrainOutput],
          apply_for_eval: Callable[[Params, BatchExample], BatchEvalPrediction],
          train_loss: Callable[[BatchExample, BatchTrainOutput], jnp.ndarray],
          eval_metrics: Mapping[str, metrics.Metric]) -> 'Model':
    """Freezes mutable arguments to `Model` to make it `jax.jit` friendly."""
    return cls(init, apply_for_train, apply_for_eval, train_loss,
               immutabledict.immutabledict(eval_metrics))


def create_model_from_haiku(
    transformed_forward_pass: hk.Transformed,
    sample_batch: BatchExample,
    train_loss: Callable[[BatchExample, BatchTrainOutput], jnp.ndarray],
    eval_metrics: Optional[Mapping[str, metrics.Metric]] = None,
    train_kwargs: Optional[Mapping[str, Any]] = None,
    eval_kwargs: Optional[Mapping[str, Any]] = None) -> Model:
  """Creates Model after applying defaults and haiku specific preprocessing.

  Args:
    transformed_forward_pass: Transformed forward pass from `hk.transform`.
    sample_batch: Example input used to determine model parameter shapes.
    train_loss: Loss function for training that outputs per example loss.
    eval_metrics: Mapping of evaluation metric names to `Metric`s. These
      `Metric`s are defined for single examples and will be consumed in
      `evaluate_model`.
    train_kwargs: Keyword arguments passed to model for training.
    eval_kwargs: Keyword arguments passed to model for evaluation.

  Returns:
    Model
  """
  eval_metrics = eval_metrics or {}
  train_kwargs = train_kwargs or {}
  eval_kwargs = eval_kwargs or {}

  @jax.jit
  def init(rng):
    return transformed_forward_pass.init(rng, sample_batch)

  @jax.jit
  def apply_for_train(params, batch, rng=None):
    return transformed_forward_pass.apply(params, rng, batch, **train_kwargs)

  @jax.jit
  def apply_for_eval(params, batch):
    return transformed_forward_pass.apply(params, None, batch, **eval_kwargs)

  return Model.new(init, apply_for_train, apply_for_eval, train_loss,
                   eval_metrics)


def create_model_from_stax(
    stax_init: Callable[..., Params],
    stax_apply: Callable[..., jnp.ndarray],
    sample_shape: Tuple[int, ...],
    train_loss: Callable[[BatchExample, BatchTrainOutput], jnp.ndarray],
    eval_metrics: Optional[Mapping[str, metrics.Metric]] = None,
    train_kwargs: Optional[Mapping[str, Any]] = None,
    eval_kwargs: Optional[Mapping[str, Any]] = None,
    input_key: str = 'x') -> Model:
  """Creates Model after applying defaults and stax specific preprocessing.

  Args:
    stax_init: Initialization function returned from `stax.serial`.
    stax_apply: Model forward_pass pass function returned from `stax.serial`.
    sample_shape: The expected shape of the input to the model.
    train_loss: Loss function for training that outputs per example loss.
    eval_metrics: Mapping of evaluation metric names to `Metric`s. These
      `Metric`s are defined for single examples and will be consumed in
      `evaluate_model`.
    train_kwargs: Keyword arguments passed to model for training.
    eval_kwargs: Keyword arguments passed to model for evaluation.
    input_key: Key name for the input in batch mapping.

  Returns:
    Model
  """
  eval_metrics = eval_metrics or {}
  train_kwargs = train_kwargs or {}
  eval_kwargs = eval_kwargs or {}

  @jax.jit
  def init(rng):
    _, params = stax_init(rng, sample_shape)
    return params

  @jax.jit
  def apply_for_train(params, batch, rng=None):
    return stax_apply(params, batch[input_key], rng=rng, **train_kwargs)

  @jax.jit
  def apply_for_eval(params, batch):
    return stax_apply(params, batch[input_key], **eval_kwargs)

  return Model.new(init, apply_for_train, apply_for_eval, train_loss,
                   eval_metrics)


@functools.partial(jax.jit, static_argnums=(0, 1))
def _evaluate_model_step(mask_key: str, model: Model, params: Params,
                         batch: BatchExample,
                         stat: metrics.Stat) -> Dict[str, metrics.Stat]:
  """Evaluates model for one batch and returns merged `Stat`.

  Args:
    mask_key: Reserved key name in example mapping for the example level mask.
    model: `Model` container with `apply_for_eval` and `eval_metrics`.
    params: Pytree of model parameters to be evaluated.
    batch: Batch of N examples.
    stat: Intermediate `Stat` from the previous step to be accumulated in the
      current step.

  Returns:
    A dictionary of intermediate evaluation `Stat`s.
  """
  try:
    mask = batch[mask_key].astype(jnp.bool_)
  except KeyError:
    mask = jnp.ones([len(next(iter(batch.values())))], dtype=jnp.bool_)
  pred = model.apply_for_eval(params, batch)
  new_stat = {
      k: metrics.evaluate_batch(metric, batch, pred, mask)
      for k, metric in model.eval_metrics.items()
  }
  return jax.tree_util.tree_multimap(
      lambda a, b: a.merge(b),
      stat,
      new_stat,
      is_leaf=lambda v: isinstance(v, metrics.Stat))


def evaluate_model(model: Model,
                   params: Params,
                   batches: Iterable[BatchExample],
                   mask_key='mask') -> Dict[str, jnp.ndarray]:
  """Evaluates model for multiple batches and returns final results.

  This is the recommended way to compute evaluation metrics for a given model.

  Args:
    model: `Model` container with `apply_for_eval` and `eval_metrics`.
    params: Pytree of model parameters to be evaluated.
    batches: Multiple batches to compute and aggregate evaluation metrics over.
    mask_key: Reserved key name in example mapping for the example level mask.

  Returns:
    A dictionary of evaluation `Metric` results.
  """
  stat = {k: metric.zero() for k, metric in model.eval_metrics.items()}
  for batch in batches:
    stat = _evaluate_model_step(mask_key, model, params, batch, stat)
  return jax.tree_util.tree_map(
      lambda x: x.result(), stat, is_leaf=lambda v: isinstance(v, metrics.Stat))
