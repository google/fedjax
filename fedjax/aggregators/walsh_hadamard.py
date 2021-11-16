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

from typing import Union

import jax
import jax.numpy as jnp
import scipy


def walsh_hadamard_transform(
    x: jnp.ndarray,
    small_n: int = 2**7,
    precision: Union[jax.lax.Precision, str] = 'highest') -> jnp.ndarray:
  """Efficient Walsh-Hadamard transform in JAX.

  An accelerator friendly O(n log n) Walsh-Hadamard transform.

  Args:
    x: A vector. len(x) must be a power of 2.
    small_n: Size to break x into. The default value is tuned on TPUv3. Must be
      a power of 2 and > 1.
    precision: Precision for general dot products.

  Returns:
    Transformed vector.
  """
  if small_n <= 1:
    raise ValueError(f'small_n must be > 1, got {small_n}')

  # Let
  # -   A ⊗ B be the Kronecker product of A and B;
  # -   flat(X) be the vector obtained by flattening the rows of X of shape
  #     [M, N].
  #
  # We can show the following:
  #
  #     (A ⊗ B^T) flat(X) = flat(A X B)
  #
  # Note that the Hadamard matrix H_{2^M 2^N} = H_{2^M} ⊗ H_{2^N}, and
  # Hadamard matrices are symmetrical. Therefore, for a [2^M, 2^N] matrix X,
  #
  #     H_{2^M 2^N} flat(X) = flat(H_{2^M} X H_{2^N})
  #
  # The idea can be generalized by breaking a Hadamard matrix into the Kronecker
  # product of many small Hadamard matrices, and reshaping the vector input into
  # a many-dimensional array, and running einsum on each dimension.
  #
  # Let the input vector be of length D, because our "small" Hadamard matrices
  # are of size at most small_n x small_n, a constant, each einsum is O(D). We
  # need to run log D einsums, thus the overall time complexity is O(D log D),
  # same as the classical divide and conquer algorithm.
  #
  # However, thanks to efficient software & hardware implementations of einsum,
  # we can often achieve far better speed than the classical algorithm on
  # accelerators, at the same time producing a far simpler XLA HLO graph.

  n = len(x)

  # Find out the shape to reshape x into.
  shape = []
  while n > 1:
    shape.append(min(n, small_n))
    n //= small_n
  shape.reverse()
  num_dims = len(shape)
  if num_dims + 1 >= 10:
    # We will run out of dimension names in einsums.
    raise ValueError(f'small_n={small_n} is too small for input size {n}')
  y = x.reshape(shape)

  # Hadamard matrices we will need.
  hadamards = dict((d, hadamard_matrix(d, x.dtype)) for d in set(shape))

  # einsum on each dimension.
  for i, d in enumerate(shape):
    y_dims = ''.join(str(j) for j in range(num_dims))
    h_dims = f'{i}{num_dims + 1}'
    out_dims = y_dims.replace(str(i), str(num_dims + 1), 1)
    operands = f'{y_dims},{h_dims}->{out_dims}'
    y = jnp.einsum(operands, y, hadamards[d], precision=precision)
  return y.flatten()


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
