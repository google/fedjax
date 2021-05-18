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
"""Toy regression models."""

from fedjax.core import models

import haiku as hk
import jax.numpy as jnp
import numpy as np


def create_regression_model() -> models.Model:
  """Creates toy regression model.

  Matches the model used in:

  Communication-Efficient Agnostic Federated Averaging
    Jae Ro, Mingqing Chen, Rajiv Mathews, Mehryar Mohri, Ananda Theertha Suresh
    https://arxiv.org/abs/2104.02748

  Returns:
    Model
  """

  def forward_pass(batch):
    network = hk.Sequential([hk.Linear(1, with_bias=False)])
    return jnp.mean(network(batch['x']))

  def train_loss(batch, preds):
    return jnp.square(jnp.mean(batch['y']) - preds)

  transformed_forward_pass = hk.transform(forward_pass)
  sample_batch = {'x': np.zeros((1, 1)), 'y': np.zeros((1,))}
  return models.create_model_from_haiku(
      transformed_forward_pass=transformed_forward_pass,
      sample_batch=sample_batch,
      train_loss=train_loss)
