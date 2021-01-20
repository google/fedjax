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
"""Toy regression models."""

import collections

from fedjax import core
import haiku as hk
import jax.numpy as jnp
import numpy as np


def create_regression_model() -> core.Model:
  """Creates toy regression model."""

  def forward_pass(batch):
    network = hk.Sequential([hk.Linear(1, with_bias=False)])
    return jnp.mean(network(batch['x']))

  def loss(batch, preds):
    return core.MeanMetric(
        total=jnp.square(jnp.mean(batch['y']) - preds), count=1.0)

  transformed_forward_pass = hk.transform(forward_pass)
  sample_batch = collections.OrderedDict(x=np.zeros((1, 1)), y=np.zeros((1,)))
  return core.create_model_from_haiku(
      transformed_forward_pass=transformed_forward_pass,
      sample_batch=sample_batch,
      loss_fn=loss)
