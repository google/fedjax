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
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Mapping, Tuple

from fedjax.core import client_datasets
from fedjax.core import dataclasses
from fedjax.core import federated_data
from fedjax.core import for_each_client
from fedjax.core import metrics
from fedjax.core import util
from fedjax.core.typing import BatchExample
from fedjax.core.typing import BatchPrediction
from fedjax.core.typing import Params
from fedjax.core.typing import PRNGKey

import haiku as hk
import jax
import jax.numpy as jnp

# Typically unnormalized model forward pass output.
BatchTrainOutput = jnp.ndarray
BatchEvalPrediction = BatchPrediction


@dataclasses.dataclass
class Model:
  """Container class for models.

  Model exists to provide easy access to predefined neural network models.
  It is meant to contain all the information needed for standard centralized
  training and evaluation. Non-standard training methods can be built upon the
  information avaiable in Model along with any additional information
  (e.g. interpolation can be implemented as a composition of two models along
  with an interpolation weight).

  Works for Haiku and jax.experimental.stax.

  The expected usage of Model is as follows::

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
    print(fedjax.evaluate_model(model, params, batches))
    # Example output:
    # {'loss': 2.3, 'accuracy': 0.2}

  The following is an example using Model compositionally as a building block
  to impelement model interpolation::

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

      return fedjax.Model(init,
                          apply_for_train,
                          apply_for_eval,
                          model_1.train_loss,
                          model_1.eval_metrics)

    model = interpolate(model_1, model_2, init_weight=0.5)

  Attributes:
    init: Initialization function that takes a seed PRNGKey and returns
      a PyTree of initialized parameters (i.e. model weights). These parameters
      will be passed as input into :meth:`apply_for_train` and
      :meth:`apply_for_eval`.
      Any trainable weights for a model that are modified in the training loop
      should be contained inside of these parameters.
    apply_for_train: Function that takes the parameters PyTree, batch of
      examples, and PRNGKey as inputs and outputs the model predictions for
      training that are then passed into :meth:`train_loss`.
      This considers strategies such as dropout.
    apply_for_eval: Function that usually takes the parameters PyTree and batch
      of examples as inputs and outputs the model predictions for evaluation
      that are then passed to :attr:`eval_metrics`.
      This is defined separately from :meth:`apply_for_train` to avoid
      having to specify inputs like PRNGKey that are not used in evaluation.
    train_loss: Loss function for training that takes batch of examples and
      model output from :meth:`apply_for_train` as input that outputs per
      example loss. This will typically called inside a :func:`jax.grad` wrapped
      function to compute gradients.
    eval_metrics: Ordered mapping of evaluation metric names to
      :class:`~fedjax.metrics.Metric`. These :class:`~fedjax.metrics.Metric` s
      are defined for single examples and will be used in :func:`evaluate_model`
  """
  init: Callable[[PRNGKey], Params]
  apply_for_train: Callable[[Params, BatchExample, PRNGKey], BatchTrainOutput]
  apply_for_eval: Callable[[Params, BatchExample], BatchEvalPrediction]
  train_loss: Callable[[BatchExample, BatchTrainOutput], jnp.ndarray]
  eval_metrics: Mapping[str, metrics.Metric]

  # Prevent dataclass from creating hash/eq so that a Model object remains
  # id hashed. This allows eval_metrics to be a standard dict.
  def __hash__(self) -> int:
    return id(self)

  def __eq__(self, other: Any) -> bool:
    return self is other


def create_model_from_haiku(
    transformed_forward_pass: hk.Transformed,
    sample_batch: BatchExample,
    train_loss: Callable[[BatchExample, BatchTrainOutput], jnp.ndarray],
    eval_metrics: Optional[Mapping[str, metrics.Metric]] = None,
    train_kwargs: Optional[Mapping[str, Any]] = None,
    eval_kwargs: Optional[Mapping[str, Any]] = None) -> Model:
  """Creates Model after applying defaults and haiku specific preprocessing.

  Args:
    transformed_forward_pass: Transformed forward pass from :func:`hk.transform`
    sample_batch: Example input used to determine model parameter shapes.
    train_loss: Loss function for training that outputs per example loss.
    eval_metrics: Mapping of evaluation metric names to
      :class:`~fedjax.metrics.Metric`. These metrics are defined for
      single examples and will be consumed in :func:`evaluate_model`.
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

  return Model(init, apply_for_train, apply_for_eval, train_loss, eval_metrics)


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
    stax_init: Initialization function returned from :func:`stax.serial`.
    stax_apply: Model forward_pass pass function returned from stax.serial.
    sample_shape: The expected shape of the input to the model.
    train_loss: Loss function for training that outputs per example loss.
    eval_metrics: Mapping of evaluation metric names to
      :class:`~fedjax.metrics.Metric`. These metrics are defined for
      single examples and will be consumed in :func:`evaluate_model`.
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

  return Model(init, apply_for_train, apply_for_eval, train_loss, eval_metrics)


@functools.partial(jax.jit, static_argnums=0)
def _evaluate_model_step(model: Model, params: Params, batch: BatchExample,
                         stat: metrics.Stat) -> Dict[str, metrics.Stat]:
  """Evaluates model for one batch and returns merged Stat.

  Args:
    model: Model container with apply_for_eval and eval_metrics.
    params: Pytree of model parameters to be evaluated.
    batch: Batch of N examples.
    stat: Intermediate Stat from the previous step to be accumulated in the
      current step.

  Returns:
    A dictionary of intermediate evaluation Stats.
  """
  try:
    mask = batch[client_datasets.EXAMPLE_MASK_KEY].astype(jnp.bool_)
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


def evaluate_model(model: Model, params: Params,
                   batches: Iterable[BatchExample]) -> Dict[str, jnp.ndarray]:
  """Evaluates model for multiple batches and returns final results.

  This is the recommended way to compute evaluation metrics for a given model.

  Args:
    model: Model container.
    params: Pytree of model parameters to be evaluated.
    batches: Multiple batches to compute and aggregate evaluation metrics over.
      Each batch can optional contain a feature keyed by
      client_datasets.MASK_KEY (see :meth:`ClientDataset.padded_batch` ).

  Returns:
    A dictionary of evaluation :class:`~fedjax.metrics.Metric` results.
  """
  stat = {k: metric.zero() for k, metric in model.eval_metrics.items()}
  for batch in batches:
    stat = _evaluate_model_step(model, params, batch, stat)
  return jax.tree_util.tree_map(
      lambda x: x.result(), stat, is_leaf=lambda v: isinstance(v, metrics.Stat))


class ModelEvaluator:
  """Evaluates model for each client dataset, either using global params, or per client params.

  To evaluate a Model on a single dataset, use evaluate_model() instead.
  """

  def __init__(self, model: Model):
    # params can be passed in 2 ways:
    # -   As `shared_input`: All clients are evaluated using the same params.
    # -   As `client_input`: Each client is evaluated using per client params.
    def client_init(shared_input, client_input):
      if shared_input is not None:
        params = shared_input
      else:
        params = client_input
      stat = {k: metric.zero() for k, metric in model.eval_metrics.items()}
      return params, stat

    def client_step(state, batch):
      params, stat = state
      next_stat = _evaluate_model_step(model, params, batch, stat)
      return params, next_stat

    def client_final(shared_input, state):
      del shared_input
      _, stat = state
      return {k: v.result() for k, v in stat.items()}

    self._evaluate_each_client = for_each_client.for_each_client(
        client_init, client_step, client_final)

  def evaluate_global_params(
      self, params: Params, clients: Iterable[Tuple[federated_data.ClientId,
                                                    Iterable[BatchExample]]]
  ) -> Iterator[Tuple[federated_data.ClientId, Dict[str, jnp.ndarray]]]:
    """Evaluates batches from each client using global params.

    Args:
      params: Model params to evaluate.
      clients: Client batches.

    Yields:
      Pairs of the client id and a dictionary of evaluation `Metric` results for
      each client.
    """
    yield from self._evaluate_each_client(
        shared_input=params,
        clients=[(client_id, batches, None) for client_id, batches in clients])

  def evaluate_per_client_params(
      self, clients: Iterable[Tuple[federated_data.ClientId,
                                    Iterable[BatchExample], Params]]
  ) -> Iterator[Tuple[federated_data.ClientId, Dict[str, jnp.ndarray]]]:
    """Evaluates batches from each client using per client params.

    Args:
      clients: Client batches and the per client params.

    Yields:
      Pairs of the client id and a dictionary of evaluation `Metric` results for
      each client.
    """
    yield from self._evaluate_each_client(shared_input=None, clients=clients)


def model_per_example_loss(
    model: Model) -> Callable[[Params, BatchExample, PRNGKey], jnp.ndarray]:
  """Convenience function for constructing a per-example loss function from a model.

  Args:
    model: Model.

  Returns:
    A function from (params, batch_example, rng) to a vector of loss values for
    each example in the batch.
  """

  def per_example_loss(params, batch_example, rng):
    train_output = model.apply_for_train(params, batch_example, rng)
    return model.train_loss(batch_example, train_output)

  return per_example_loss


@functools.partial(jax.jit, static_argnums=0)
def _evaluate_average_loss_step(per_example_loss, params, batch, rng,
                                accum_loss, num_examples):
  """Evaluates average per example loss for one batch and returns updated accumlators."""
  rng, use_rng = jax.random.split(rng)
  loss = per_example_loss(params, batch, use_rng)
  if client_datasets.EXAMPLE_MASK_KEY in batch:
    mask = batch[client_datasets.EXAMPLE_MASK_KEY]
    accum_loss += jnp.vdot(mask, loss)
    num_examples += jnp.sum(mask)
  else:
    accum_loss += jnp.sum(loss)
    num_examples += len(loss)
  return rng, accum_loss, num_examples


@functools.partial(jax.jit, static_argnums=0)
def _finalize_average_loss(regularizer, params, accum_loss, num_examples):
  average_loss = util.safe_div(accum_loss, num_examples)
  if regularizer is not None:
    average_loss += regularizer(params)
  return average_loss


def evaluate_average_loss(
    params: Params,
    batches: Iterable[BatchExample],
    rng: PRNGKey,
    per_example_loss: Callable[[Params, BatchExample, PRNGKey], jnp.ndarray],
    regularizer: Optional[Callable[[Params],
                                   jnp.ndarray]] = None) -> jnp.ndarray:
  """Evaluates the average per example loss over multiple batches.

  Args:
    params: PyTree of model parameters to be evaluated.
    batches: Multiple batches to compute and aggregate evaluation metrics over.
      Each batch can optional contain a feature keyed by
      client_datasets.MASK_KEY (see ClientDataset.padded_batch).
    rng: Initial PRNGKey for making per_example_loss calls.
    per_example_loss: Per example loss function.
    regularizer: Optional regularizer function.

  Returns:
    The average per example loss, plus the regularizer term when specified.
  """
  accum_loss, num_examples = 0, 0
  for batch in batches:
    rng, accum_loss, num_examples = _evaluate_average_loss_step(
        per_example_loss=per_example_loss,
        params=params,
        batch=batch,
        rng=rng,
        accum_loss=accum_loss,
        num_examples=num_examples)
  return _finalize_average_loss(
      regularizer=regularizer,
      params=params,
      accum_loss=accum_loss,
      num_examples=num_examples)


class AverageLossEvaluator:
  """Evaluates average loss for each client dataset, either using global params, or per client params.

  The average loss is defined as the average per example loss, plus the
  regularizer term when specified. To evaluate average loss on a single dataset,
  use evaluate_average_loss() instead.
  """

  def __init__(self,
               per_example_loss: Callable[[Params, BatchExample, PRNGKey],
                                          jnp.ndarray],
               regularizer: Optional[Callable[[Params], jnp.ndarray]] = None):
    # params can be passed in 2 ways:
    # -   As `shared_input`: All clients are evaluated using the same params.
    # -   As `client_input`: Each client is evaluated using per client params.
    def client_init(shared_input, client_input):
      if shared_input is not None:
        params = shared_input
        rng = client_input
      else:
        rng, params = client_input
      accum_loss = 0.
      num_examples = 0.
      return rng, params, accum_loss, num_examples

    def client_step(state, batch):
      rng, params, accum_loss, num_examples = state
      rng, accum_loss, num_examples = _evaluate_average_loss_step(
          per_example_loss=per_example_loss,
          params=params,
          batch=batch,
          rng=rng,
          accum_loss=accum_loss,
          num_examples=num_examples)
      return rng, params, accum_loss, num_examples

    def client_final(shared_input, state):
      del shared_input
      _, params, accum_loss, num_examples = state
      return _finalize_average_loss(
          regularizer=regularizer,
          params=params,
          accum_loss=accum_loss,
          num_examples=num_examples)

    self._evaluate_each_client = for_each_client.for_each_client(
        client_init, client_step, client_final)

  def evaluate_global_params(
      self, params: Params, clients: Iterable[Tuple[federated_data.ClientId,
                                                    Iterable[BatchExample],
                                                    PRNGKey]]
  ) -> Iterator[Tuple[federated_data.ClientId, jnp.ndarray]]:
    """Evaluates batches from each client using global params.

    Args:
      params: Model params to evaluate.
      clients: Client batches.

    Yields:
      Pairs of the client id and the client's average loss.
    """
    yield from self._evaluate_each_client(shared_input=params, clients=clients)

  def evaluate_per_client_params(
      self, clients: Iterable[Tuple[federated_data.ClientId,
                                    Iterable[BatchExample], PRNGKey, Params]]
  ) -> Iterator[Tuple[federated_data.ClientId, jnp.ndarray]]:
    """Evaluates batches from each client using per client params.

    Args:
      clients: Client batches and the per client params.

    Yields:
      Pairs of the client id and the client's average loss.
    """
    yield from self._evaluate_each_client(
        shared_input=None,
        clients=[(client_id, batches, (rng, params))
                 for client_id, batches, rng, params in clients])


def grad(
    per_example_loss: Callable[[Params, BatchExample, PRNGKey], jnp.ndarray],
    regularizer: Optional[Callable[[Params], jnp.ndarray]] = None
) -> Callable[[Params, BatchExample, PRNGKey], Params]:
  """A standard gradient function derived from per-example loss and an optional regularizer.

  The scalar loss function being differentiated is simply:

    mean(per-example loss) + regularizer term

  The returned gradient function support both unpadded batches, and padded
  batches with the mask feature keyed by client_datasets.EXAMPLE_MASK_KEY.

  Args:
    per_example_loss: A function from (params, batch_example, rng) to a vector
      of loss values for each example in the batch.
    regularizer: Optional regularizer that only depends on params.

  Returns:
    A function from (params, batch_example, rng) to gradients.
  """

  def scalar_loss(params, batch_example, rng):
    batch_loss = per_example_loss(params, batch_example, rng)
    if client_datasets.EXAMPLE_MASK_KEY in batch_example:
      mask = batch_example[client_datasets.EXAMPLE_MASK_KEY]
      num_examples = jnp.sum(mask)
      loss = util.safe_div(jnp.vdot(batch_loss, mask), num_examples)
    else:
      loss = jnp.mean(batch_loss)
    if regularizer is not None:
      loss += regularizer(params)
    return loss

  return jax.jit(jax.grad(scalar_loss))


def model_grad(
    model: Model,
    regularizer: Optional[Callable[[Params], jnp.ndarray]] = None
) -> Callable[[Params, BatchExample, PRNGKey], Params]:
  """A standard gradient function derived from a model and an optional regularizer.

  The scalar loss function being differentiated is simply:

    mean(model's per-example loss) + regularizer term

  The returned gradient function support both unpadded batches, and padded
  batches with the mask feature keyed by client_datasets.EXAMPLE_MASK_KEY.

  Args:
    model: A Model.
    regularizer: Optional regularizer.

  Returns:
    A function from (params, batch_example, rng) to gradients.
  """

  return grad(model_per_example_loss(model), regularizer)
