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
"""Efficient Walsh-Hadamard transform in JAX."""

import functools

import jax
import jax.numpy as jnp
import scipy


def walsh_hadamard_transform(x: jnp.ndarray,
                             small_n: int = 2**9,
                             medium_n: int = 2**13) -> jnp.ndarray:
  """Efficient Walsh-Hadamard transform in JAX.

  The actual implementation is selected based on input size. The default
  thresholds are tuned on TPUv3.

  * When len(x) <= small_n, uses naive_walsh_hadamard_transform().
  * When small_n < len(x) <= medium_n, uses
    top_down_fast_walsh_hadamard_transform().
  * Otherwise, uses bottom_up_fast_walsh_hadamard_transform().

  Args:
    x: A vector. len(x) must be a power of 2.
    small_n: Input size threshold.
    medium_n: Input size threshold.

  Returns:
    Transformed vector.
  """
  n = len(x)
  if n <= small_n:
    return naive_walsh_hadamard_transform(x)
  elif n <= medium_n:
    return top_down_fast_walsh_hadamard_transform(x)
  else:
    return bottom_up_fast_walsh_hadamard_transform(x)


def hadamard_matrix(n: int, dtype: jnp.dtype) -> jnp.ndarray:
  """Generates the Hadamard matrix.

  Because there are JAX dtypes not supported in numpy, the equivalent function
  in scipy can't be used directly.

  Args:
    n: Number of rows/columns of the Hadamard matrix. Must be a power of 2.
    dtype: Output dtype.

  Returns:
    The Hadamard matrix of the given size and type.
  """
  return jnp.array(scipy.linalg.hadamard(n), dtype)


# Below we use the highest precision for dot products. This is necessary for
# obtaining accurate results on TPUs. Benchmarks have shown negligible speed
# difference between the default precision level and the highest.


@jax.jit
def naive_walsh_hadamard_transform(x: jnp.ndarray) -> jnp.ndarray:
  """Walsh-Hadamard transform as direct matrix multiplication.

  Suitable for small inputs.

  Args:
    x: A vector. len(x) must be a power of 2.

  Returns:
    Transformed vector.
  """
  return jnp.dot(
      hadamard_matrix(len(x), x.dtype), x, precision=jax.lax.Precision.HIGHEST)


@functools.partial(jax.jit, static_argnums=1)
def top_down_fast_walsh_hadamard_transform(x: jnp.ndarray,
                                           small_n: int = 512) -> jnp.ndarray:
  """Fast Walsh-Hadamard transform in a top-down implementation.

  Suitable for medium sized inputs.

  Args:
    x: A vector. len(x) must be a power of 2.
    small_n: Input size threshold for falling back to
      naive_walsh_hadamard_transform().

  Returns:
    Transformed vector.
  """
  n = len(x)
  if n <= small_n:
    return naive_walsh_hadamard_transform(x)

  h_small = hadamard_matrix(small_n, x.dtype)

  def transform(x):
    n = len(x)
    assert n >= small_n
    if n == small_n:
      return jnp.dot(h_small, x, precision=jax.lax.Precision.HIGHEST)
    else:
      x_l, x_h = jnp.split(x, 2)
      y_l = transform(x_l)
      y_h = transform(x_h)
      return jnp.concatenate([y_l + y_h, y_l - y_h])

  return transform(x)


@functools.partial(jax.jit, static_argnums=1)
def bottom_up_fast_walsh_hadamard_transform(x: jnp.ndarray,
                                            small_n: int = 512) -> jnp.ndarray:
  """Fast Walsh-Hadamard transform in a bottom-up implementation.

  Suitable for large inputs.

  Args:
    x: A vector. len(x) must be a power of 2.
    small_n: Input size threshold for falling back to
      naive_walsh_hadamard_transform().

  Returns:
    Transformed vector.
  """
  n = len(x)
  if n <= small_n:
    return naive_walsh_hadamard_transform(x)

  h_small = hadamard_matrix(small_n, x.dtype)
  x_small = x.reshape([-1, 2, small_n])
  # [n/2/small_n, 2 * small_n]
  y_lh = jnp.einsum(
      'ij,klj->kli', h_small, x_small,
      precision=jax.lax.Precision.HIGHEST).reshape([-1, 2 * small_n])
  i = small_n
  while i < n:
    # Invariant: y_lh is [n/2/i, 2*i]
    y_l, y_h = jnp.split(y_lh, 2, axis=-1)
    # [n/2/i, 2*i]
    y = jnp.concatenate([y_l + y_h, y_l - y_h], axis=-1)
    i *= 2
    if i == n:
      return y[0]
    y_lh = y.reshape([-1, 2 * i])
