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

from fedjax.experimental.typing import PyTree

import jax
import jax.numpy as jnp


@jax.jit
def tree_size(pytree: PyTree) -> int:
  return sum(l.size for l in jax.tree_util.tree_leaves(pytree))


@jax.jit
def tree_l2_norm(pytree: PyTree) -> float:
  return jnp.sqrt(
      sum(jnp.vdot(x, x) for x in jax.tree_util.tree_leaves(pytree)))
