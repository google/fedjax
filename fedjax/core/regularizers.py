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
"""Library for defining regularizers."""

import abc
from typing import Optional

from fedjax.core.typing import Params
import jax
import jax.numpy as jnp


class Regularizer(metaclass=abc.ABCMeta):
  """Base class representing regularizers."""

  def __init__(self, center_params: Optional[Params]):
    self._center_params = center_params

    def center_fn(params):
      return jax.tree_multimap(lambda a, b: a - b, params, center_params)

    if center_params is not None:
      self._preprocess_fn = center_fn
    else:
      self._preprocess_fn = lambda p: p

  @abc.abstractmethod
  def __call__(self, params: Params) -> float:
    """Evaluates the regularizer."""


class L2Regularizer(Regularizer):
  """Class representing an L2 regularizer.

  Attributes:
    _center_params: Param values to regularize toward.
    _weight: Weight in front of L2 norm.
    _param_weights: Optional per-parameter weighting, which allows different
      regularization strength for each parameter.
  """

  def __init__(self,
               center_params: Optional[Params] = None,
               weight: float = 1.0,
               param_weights: Optional[Params] = None):
    super().__init__(center_params)
    self._weight = weight
    self._param_weights = param_weights

  def __call__(self, params: Params) -> float:
    """Evaluates the regularizer."""
    params = self._preprocess_fn(params)
    leaves, _ = jax.tree_flatten(params)

    if self._param_weights:
      param_weight_leaves, _ = jax.tree_flatten(self._param_weights)
      return sum(
          jnp.vdot(pw, jnp.square(x))
          for pw, x in zip(param_weight_leaves, leaves)) * self._weight

    return sum(jnp.vdot(x, x) for x in leaves) * self._weight
