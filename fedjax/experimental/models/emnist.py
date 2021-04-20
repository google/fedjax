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
"""EMNIST models."""

from fedjax.experimental import metrics
from fedjax.experimental import model
from fedjax.models import emnist

import haiku as hk
import jax
from jax.experimental import stax
import numpy as np


# Defines the expected structure of input batches to the model. This is used to
# determine the model parameter shapes.
_HAIKU_SAMPLE_BATCH = {'x': np.zeros((1, 28, 28, 1)), 'y': np.zeros(1,)}
_STAX_SAMPLE_SHAPE = (-1, 28, 28, 1)

_TRAIN_LOSS = lambda b, p: metrics.unreduced_cross_entropy_loss(b['y'], p)
_EVAL_METRICS = {
    'loss': metrics.CrossEntropyLoss(),
    'accuracy': metrics.Accuracy()
}


def create_conv_model(only_digits: bool = False) -> model.Model:
  """Creates EMNIST CNN model with dropout with haiku.

  Matches the model used in:

  Adaptive Federated Optimization
    Sashank Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush,
    Jakub Konečný, Sanjiv Kumar, H. Brendan McMahan.
    https://arxiv.org/abs/2003.00295

  Args:
    only_digits: Whether to use only digit classes [0-9] or include lower and
      upper case characters for a total of 62 classes.

  Returns:
    Model
  """
  num_classes = 10 if only_digits else 62

  def forward_pass(batch, is_train=True):
    return emnist.ConvDropoutModule(num_classes)(batch['x'], is_train)

  transformed_forward_pass = hk.transform(forward_pass)
  return model.create_model_from_haiku(
      transformed_forward_pass=transformed_forward_pass,
      sample_batch=_HAIKU_SAMPLE_BATCH,
      train_loss=_TRAIN_LOSS,
      eval_metrics=_EVAL_METRICS,
      # is_train determines whether to apply dropout or not.
      train_kwargs={'is_train': True},
      eval_kwargs={'is_train': False})


def create_dense_model(only_digits: bool = False,
                       hidden_units: int = 200) -> model.Model:
  """Creates EMNIST dense net with haiku."""
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
  return model.create_model_from_haiku(
      transformed_forward_pass=transformed_forward_pass,
      sample_batch=_HAIKU_SAMPLE_BATCH,
      train_loss=_TRAIN_LOSS,
      eval_metrics=_EVAL_METRICS)


def create_logistic_model(only_digits: bool = False) -> model.Model:
  """Creates EMNIST logistic model with haiku."""
  num_classes = 10 if only_digits else 62

  def forward_pass(batch):
    network = hk.Sequential([
        hk.Flatten(),
        hk.Linear(num_classes),
    ])
    return network(batch['x'])

  transformed_forward_pass = hk.transform(forward_pass)
  return model.create_model_from_haiku(
      transformed_forward_pass=transformed_forward_pass,
      sample_batch=_HAIKU_SAMPLE_BATCH,
      train_loss=_TRAIN_LOSS,
      eval_metrics=_EVAL_METRICS)


def create_stax_dense_model(only_digits: bool = False,
                            hidden_units: int = 200) -> model.Model:
  """Creates EMNIST dense net with stax."""
  num_classes = 10 if only_digits else 62
  stax_init, stax_apply = stax.serial(stax.Flatten,
                                      stax.Dense(hidden_units), stax.Relu,
                                      stax.Dense(hidden_units), stax.Relu,
                                      stax.Dense(num_classes))
  return model.create_model_from_stax(
      stax_init=stax_init,
      stax_apply=stax_apply,
      sample_shape=_STAX_SAMPLE_SHAPE,
      train_loss=_TRAIN_LOSS,
      eval_metrics=_EVAL_METRICS)
