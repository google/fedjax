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
"""Utilities for working with tree-like container data structures."""

from typing import Iterable, List, Tuple, TypeVar

import jax
import jax.numpy as jnp

T = TypeVar('T')
tree_map = jax.tree_util.tree_map
tree_multimap = jax.tree_util.tree_multimap


@jax.jit
def tree_stack(pytrees: List[T]) -> T:
  """Stacks input pytrees along leading axis into a single pytree."""
  return jax.tree_multimap(lambda *args: jnp.stack(args), *pytrees)


@jax.jit
def tree_unstack(pytree: T) -> List[T]:
  """Splits pytree along leading axis into a list of pytrees."""
  leaves, treedef = jax.tree_flatten(pytree)
  n = leaves[0].shape[0]
  split_leaves = [[] for _ in range(n)]
  for l in leaves:
    for i, sl in enumerate(jnp.vsplit(l, n)):
      # Squeeze off additional axis left from split.
      split_leaves[i].append(jnp.squeeze(sl, axis=0))
  return [jax.tree_unflatten(treedef, sl) for sl in split_leaves]


@jax.jit
def tree_weight(pytree: T, weight: float) -> T:
  """Weights tree leaves by weight."""
  return jax.tree_map(lambda l: l * weight, pytree)


@jax.jit
def tree_zeros_like(pytree: T) -> T:
  """Creates a tree with zeros with same structure as the input."""
  return jax.tree_map(jnp.zeros_like, pytree)


@jax.jit
def tree_sum(*pytrees: Iterable[T]) -> T:
  """Sums input trees together."""
  sum_tree = pytrees[0]
  for tree in pytrees[1:]:
    sum_tree = jax.tree_multimap(lambda a, b: a + b, sum_tree, tree)
  return sum_tree


def tree_mean(pytrees_and_weights: Iterable[Tuple[T, float]]) -> T:
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
      sum_weighted_pytree = tree_sum(sum_weighted_pytree, weighted_pytree)
    sum_weight += weight
  inverse_weight = (1. / sum_weight) if sum_weight > 0. else 0.
  return tree_weight(sum_weighted_pytree, inverse_weight)
