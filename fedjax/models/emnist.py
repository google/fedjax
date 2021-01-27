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
"""EMNIST models."""

import collections
from typing import Callable, Optional

from fedjax import core
import haiku as hk
import jax
from jax.experimental import stax
import jax.numpy as jnp
import numpy as np


class Dropout(hk.Module):
  """Dropout haiku module."""

  def __init__(self, rate: float = 0.5):
    """Initializes dropout module.

    Args:
      rate: Probability that each element of x is discarded. Must be in [0, 1).
    """
    super().__init__()
    self._rate = rate

  def __call__(self, x: jnp.ndarray, is_train: bool):
    if is_train:
      return hk.dropout(rng=hk.next_rng_key(), rate=self._rate, x=x)
    return x


class ConvDropoutModule(hk.Module):
  """Custom haiku module for CNN with dropout.

  This must be defined as a custom hk.Module because only a single positional
  argument is allowed when using hk.Sequential.
  """

  def __init__(self, num_classes):
    super().__init__()
    self._num_classes = num_classes

  def __call__(self, x: jnp.ndarray, is_train: bool):
    x = hk.Conv2D(output_channels=32, kernel_shape=(3, 3), padding='VALID')(x)
    x = jax.nn.relu(x)
    x = hk.Conv2D(output_channels=64, kernel_shape=(3, 3), padding='VALID')(x)
    x = jax.nn.relu(x)
    x = (
        hk.MaxPool(
            window_shape=(1, 2, 2, 1), strides=(1, 2, 2, 1),
            padding='VALID')(x))
    x = Dropout(rate=0.25)(x, is_train)
    x = hk.Flatten()(x)
    x = hk.Linear(128)(x)
    x = jax.nn.relu(x)
    x = Dropout(rate=0.5)(x, is_train)
    x = hk.Linear(self._num_classes)(x)
    return x


# Defines the expected structure of input batches to the model. This is used to
# determine the model parameter shapes.
_EMNIST_HAIKU_SAMPLE_BATCH = collections.OrderedDict(
    x=np.zeros((1, 28, 28, 1)), y=np.zeros(1,))
_EMNIST_STAX_SAMPLE_SHAPE = (-1, 28, 28, 1)


def _loss(batch: core.Batch, preds: jnp.ndarray) -> core.Metric:
  return core.metrics.cross_entropy_loss_fn(targets=batch['y'], preds=preds)


def _accuracy(batch: core.Batch, preds: jnp.ndarray) -> core.Metric:
  return core.metrics.accuracy_fn(targets=batch['y'], preds=preds)


# Common definitions for EMNIST image recognition task.
_EMNIST_LOSS_FN = _loss
_EMNIST_METRICS_FN_MAP = collections.OrderedDict(accuracy=_accuracy)


def create_conv_model(
    only_digits: bool = False,
    reg_fn: Optional[Callable[[core.Params],
                              jnp.ndarray]] = None) -> core.Model:
  """Creates EMNIST CNN model with dropout."""
  num_classes = 10 if only_digits else 62

  def forward_pass(batch, is_train=True):
    return ConvDropoutModule(num_classes)(batch['x'], is_train)

  transformed_forward_pass = hk.transform(forward_pass)
  return core.create_model_from_haiku(
      transformed_forward_pass=transformed_forward_pass,
      sample_batch=_EMNIST_HAIKU_SAMPLE_BATCH,
      loss_fn=_EMNIST_LOSS_FN,
      reg_fn=reg_fn,
      metrics_fn_map=_EMNIST_METRICS_FN_MAP,
      train_kwargs={'is_train': True},
      test_kwargs={'is_train': False})


def create_dense_model(
    only_digits: bool = False,
    hidden_units: int = 200,
    reg_fn: Optional[Callable[[core.Params],
                              jnp.ndarray]] = None) -> core.Model:
  """Creates EMNIST dense net."""
  num_classes = 10 if only_digits else 62

  def forward_pass(batch):
    network = hk.Sequential([
        hk.Flatten(),
        hk.Linear(hidden_units),
        jax.nn.relu,
        hk.Linear(hidden_units),
        jax.nn.relu,
        hk.Linear(num_classes),
    ])
    return network(batch['x'])

  transformed_forward_pass = hk.transform(forward_pass)
  return core.create_model_from_haiku(
      transformed_forward_pass=transformed_forward_pass,
      sample_batch=_EMNIST_HAIKU_SAMPLE_BATCH,
      loss_fn=_EMNIST_LOSS_FN,
      reg_fn=reg_fn,
      metrics_fn_map=_EMNIST_METRICS_FN_MAP)


def create_logistic_model(
    only_digits: bool = False,
    reg_fn: Optional[Callable[[core.Params],
                              jnp.ndarray]] = None) -> core.Model:
  """Creates EMNIST logistic model."""
  num_classes = 10 if only_digits else 62

  def forward_pass(batch):
    network = hk.Sequential([
        hk.Flatten(),
        hk.Linear(num_classes),
    ])
    return network(batch['x'])

  transformed_forward_pass = hk.transform(forward_pass)
  return core.create_model_from_haiku(
      transformed_forward_pass=transformed_forward_pass,
      sample_batch=_EMNIST_HAIKU_SAMPLE_BATCH,
      loss_fn=_EMNIST_LOSS_FN,
      reg_fn=reg_fn,
      metrics_fn_map=_EMNIST_METRICS_FN_MAP)


def create_stax_dense_model(
    only_digits: bool = False,
    hidden_units: int = 200,
    reg_fn: Optional[Callable[[core.Params],
                              jnp.ndarray]] = None) -> core.Model:
  """Creates EMNIST dense net via stax."""
  num_classes = 10 if only_digits else 62
  stax_init_fn, stax_apply_fn = stax.serial(stax.Flatten,
                                            stax.Dense(hidden_units), stax.Relu,
                                            stax.Dense(hidden_units), stax.Relu,
                                            stax.Dense(num_classes))
  return core.create_model_from_stax(
      stax_init_fn=stax_init_fn,
      stax_apply_fn=stax_apply_fn,
      sample_shape=_EMNIST_STAX_SAMPLE_SHAPE,
      loss_fn=_EMNIST_LOSS_FN,
      reg_fn=reg_fn,
      metrics_fn_map=_EMNIST_METRICS_FN_MAP)
