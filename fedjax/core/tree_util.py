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
"""Utilities for working with tree-like container data structures.

In JAX, the term pytree refers to a tree-like structure built out of
container-like Python objects.
For more details, see https://jax.readthedocs.io/en/latest/pytrees.html.
"""

from typing import Iterable, Tuple

from fedjax.core.typing import PyTree

import jax
import jax.numpy as jnp


@jax.jit
def tree_weight(pytree: PyTree, weight: float) -> PyTree:
  """Weights tree leaves by weight."""
  return jax.tree_map(lambda l: l * weight, pytree)


def tree_inverse_weight(pytree: PyTree, weight: float) -> PyTree:
  """Weights tree leaves by ``1 / weight``."""
  inverse_weight = (1. / weight) if weight > 0. else 0.
  return tree_weight(pytree, inverse_weight)


@jax.jit
def tree_zeros_like(pytree: PyTree) -> PyTree:
  """Creates a tree with zeros with same structure as the input."""
  return jax.tree_map(jnp.zeros_like, pytree)


@jax.jit
def tree_add(left: PyTree, right: PyTree) -> PyTree:
  """Adds two trees together."""
  return jax.tree_util.tree_map(jnp.add, left, right)


def tree_sum(pytrees: Iterable[PyTree]) -> PyTree:
  """Sums multiple trees together."""
  pytree_sum = None
  for pytree in pytrees:
    if pytree_sum is None:
      pytree_sum = pytree
    else:
      pytree_sum = tree_add(pytree_sum, pytree)
  return pytree_sum


def tree_mean(pytrees_and_weights: Iterable[Tuple[PyTree, float]]) -> PyTree:
  """Returns (weighted) mean of input trees and weights.

  Args:
    pytrees_and_weights: Iterable of tuples of pytrees and associated weights.

  Returns:
    (Weighted) mean of input trees and weights.
  """
  sum_weighted_pytree = None
  sum_weight = 0.
  for pytree, weight in pytrees_and_weights:
    weighted_pytree = tree_weight(pytree, weight)
    if sum_weighted_pytree is None:
      sum_weighted_pytree = weighted_pytree
    else:
      sum_weighted_pytree = tree_add(sum_weighted_pytree, weighted_pytree)
    sum_weight += weight
  return tree_inverse_weight(sum_weighted_pytree, sum_weight)


@jax.jit
def tree_size(pytree: PyTree) -> int:
  """Returns total size of all tree leaves."""
  return sum(l.size for l in jax.tree_util.tree_leaves(pytree))


@jax.jit
def tree_l2_squared(pytree: PyTree) -> float:
  """Returns squared l2 norm of tree."""
  return sum(jnp.vdot(x, x) for x in jax.tree_util.tree_leaves(pytree))


@jax.jit
def tree_l2_norm(pytree: PyTree) -> float:
  """Returns l2 norm of tree."""
  return jnp.sqrt(tree_l2_squared(pytree))


@jax.jit
def tree_clip_by_global_norm(pytree: PyTree, max_norm: float) -> PyTree:
  """Clips a pytree of arrays using their global norm.

  References:
    [Pascanu et al, 2012](https://arxiv.org/abs/1211.5063)

  Args:
    pytree: A pytree to be potentially clipped.
    max_norm: The maximum global norm for a pytree.

  Returns:
    A potentially clipped pytree.
  """
  global_norm = tree_l2_norm(pytree)
  scale = jnp.minimum(1, max_norm / global_norm)
  return jax.tree_util.tree_map(lambda t: scale * t, pytree)
