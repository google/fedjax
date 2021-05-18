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
"""Library for defining regularizers."""

from typing import Callable, Optional

from fedjax.core.typing import Params

import jax
import jax.numpy as jnp


def _l2_regularize(params: Params, weight: float,
                   center_params: Optional[Params],
                   params_weights: Optional[Params]) -> float:
  """Returns L2 regularization weight."""
  if center_params is not None:
    params = jax.tree_multimap(lambda a, b: a - b, params, center_params)
  leaves = jax.tree_util.tree_leaves(params)
  if params_weights is not None:
    pw_leaves = jax.tree_util.tree_leaves(params_weights)
    return sum(jnp.vdot(pw, jnp.square(x))
               for pw, x in zip(pw_leaves, leaves)) * weight
  return sum(jnp.vdot(x, x) for x in leaves) * weight


def l2_regularizer(
    weight: float,
    center_params: Optional[Params] = None,
    params_weights: Optional[Params] = None) -> Callable[[Params], float]:
  """Returns L2 regularization function.

  Args:
    weight: Weight applied to L2 norm.
    center_params: Model parameter values to regularize toward.
    params_weights: Per-parameter weighting, which allows different
      regularization strengths for each parameter.

  Returns:
    L2 regularization function typically used in calculating training loss.
  """

  def func(params):
    return _l2_regularize(params, weight, center_params, params_weights)

  return func
