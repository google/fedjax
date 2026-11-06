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
"""Algorithms for model update compression."""

from functools import partial
import itertools
import math
from typing import Iterable, Optional, Tuple

from fedjax.aggregators import aggregator
from fedjax.aggregators import walsh_hadamard
from fedjax.core import dataclasses
from fedjax.core import tree_util
from fedjax.core.federated_data import ClientId
from fedjax.core.typing import PRNGKey, Params

import haiku as hk
import jax
import jax.numpy as jnp


@dataclasses.dataclass
class CompressionState:
  """State of compression aggregator passed between rounds.

  Attributes:
    num_bits: number of bits transmitted per client.
    rng: random key used for compression.
  """
  num_bits: float
  rng: PRNGKey


def binary_stochastic_quantize(v: jnp.ndarray,
                               rng: PRNGKey,
                               v_min: Optional[float] = None,
                               v_max: Optional[float] = None) -> jnp.ndarray:
  """Binary stochastic algorithm in https://arxiv.org/pdf/1611.00429.pdf.

  Args:
    v: vector to be quantized.
    rng: jax random key.
    v_min: minimum threshold for quantization. If None, sets it to jnp.amin(v).
    v_max: maximum threshold for quantization. If None, sets it to jnp.amax(v).

  Returns:
    Quantized array.
  """
  if v_min is None:
    v_min = jnp.amin(v)  # pyrefly: ignore[bad-assignment]
  if v_max is None:
    v_max = jnp.amax(v)  # pyrefly: ignore[bad-assignment]
  v = jnp.nan_to_num((v - v_min) / (v_max - v_min))  # pyrefly: ignore[unsupported-operation]
  v = jnp.maximum(0., jnp.minimum(v, 1.))
  rand = jax.random.uniform(key=rng, shape=v.shape)
  return jnp.where(rand > v, v_min, v_max)  # pyrefly: ignore[bad-argument-type]


def uniform_stochastic_quantize(v: jnp.ndarray,
                                num_levels: int,
                                rng: PRNGKey,
                                v_min: Optional[float] = None,
                                v_max: Optional[float] = None) -> jnp.ndarray:
  """Uniform stochastic algorithm in https://arxiv.org/pdf/1611.00429.pdf.

  Args:
    v: vector to be quantized.
    num_levels: Number of levels of quantization.
    rng: jax random key.
    v_min: minimum threshold for quantization. If None, sets it to jnp.amin(v).
    v_max: maximum threshold for quantization. If None, sets it to jnp.amax(v).

  Returns:
    Quantized array.
  """
  # Rescale the vector to be between zero to one.
  if v_min is None:
    v_min = jnp.amin(v)  # pyrefly: ignore[bad-assignment]
  if v_max is None:
    v_max = jnp.amax(v)  # pyrefly: ignore[bad-assignment]
  v = jnp.nan_to_num((v - v_min) / (v_max - v_min))  # pyrefly: ignore[unsupported-operation]
  v = jnp.maximum(0., jnp.minimum(v, 1.))
  # Compute the upper and lower boundary of each value.
  v_ceil = jnp.ceil(v * (num_levels - 1)) / (num_levels - 1)
  v_floor = jnp.floor(v * (num_levels - 1)) / (num_levels - 1)
  # uniformly quantize between v_ceil and v_floor.
  rand = jax.random.uniform(key=rng, shape=v.shape)
  threshold = jnp.nan_to_num((v - v_floor) / (v_ceil - v_floor))
  quantized = jnp.where(rand > threshold, v_floor, v_ceil)
  # Rescale the values and return it.
  return v_min + quantized * (v_max - v_min)  # pyrefly: ignore[unsupported-operation]


@jax.jit
def uniform_stochastic_quantize_pytree(params: Params, num_levels: int,
                                       rng: PRNGKey) -> Params:
  """Applies uniform_stochastic_quantize for all leafs of the pytree.

  Args:
    params: pytree to be quantized.
    num_levels: Number of levels of quantization.
    rng: jax random key.

  Returns:
    Quantized pytree.
  """
  leaves, tree_def = jax.tree_util.tree_flatten(params)
  rngs = jax.random.split(rng, len(leaves))
  new_leaves = []
  for l, r in zip(leaves, rngs):
    new_leaves.append(uniform_stochastic_quantize(l, num_levels, r))
  return jax.tree_util.tree_unflatten(tree_def, new_leaves)


def num_leaves(pytree):
  return len(jax.tree_util.tree_leaves(pytree))


@jax.jit
def _entropy(v, uniq):
  uniq = jnp.concatenate([uniq, jnp.array([jnp.inf])], axis=0)
  hist, _ = jnp.histogram(v, bins=uniq)
  hist = hist / jnp.sum(hist)
  entropy = -jnp.sum(hist * jnp.log2(hist))
  return entropy


@jax.jit
def _hist_bits(v, uniq):
  """Number of bits required to encode the histogram of v."""
  d = v.size
  k = uniq.size
  return k * jnp.log2(jnp.exp(1)*(d + k)/k)


def arithmetic_encoding_num_bits(v: jnp.ndarray) -> int:
  """Computes number of bits needed to store v via arithmetic coding."""
  v = jnp.nan_to_num(v)
  v = v.flatten()
  uniq = jnp.unique(v)
  entropy = _entropy(v, uniq)
  hist_bits = _hist_bits(v, uniq)
  return hist_bits + (v.size * entropy) + (2 * 32) + 2


def uniform_stochastic_quantizer(
    num_levels: int,
    rng: PRNGKey,
    encode_algorithm: Optional[str] = None) -> aggregator.Aggregator:
  """Returns (weighted) mean of input uniformly quantized trees using the

  uniform stochastic algorithm in https://arxiv.org/pdf/1611.00429.pdf.

  Args:
    num_levels: number of levels of quantization.
    rng: PRNGKey used for compression:
    encode_algorithm: None or arithmetic

  Returns:
    Compression aggregator.
  """

  def init():
    return CompressionState(0.0, rng)  # pyrefly: ignore[bad-argument-count]

  def apply(
      clients_params_and_weights: Iterable[Tuple[ClientId, Params, float]],
      aggregator_state: CompressionState) -> Tuple[Params, CompressionState]:

    if encode_algorithm is not None:
      assert encode_algorithm == 'arithmetic'

    def quantize_params_and_weight(client_params_and_weight, rng):
      _, params, weight = client_params_and_weight
      return uniform_stochastic_quantize_pytree(params, num_levels, rng), weight

    rng, use_rng = jax.random.split(aggregator_state.rng)
    # TODO(theertha): remove the usage of hk.PRNGSequence.
    rng_seq = hk.PRNGSequence(use_rng)
    clients_params_and_weight_rng = zip(clients_params_and_weights, rng_seq)
    quantized_p_and_w = itertools.starmap(quantize_params_and_weight,
                                          clients_params_and_weight_rng)
    new_bits = 0.
    if encode_algorithm == 'arithmetic':
      # Accumulate the number of bits used by all clients without loading the
      # entire iterator into memory at once.
      total_bits = []

      def arithmetic_encoding_num_bits_pytree(params, weights):
        leaves, _ = jax.tree_util.tree_flatten(params)
        bits = sum([arithmetic_encoding_num_bits(leaf) for leaf in leaves])
        total_bits.append(bits)
        return params, weights

      quantized_p_and_w = itertools.starmap(arithmetic_encoding_num_bits_pytree,
                                            quantized_p_and_w)
      aggregated_params = tree_util.tree_mean(quantized_p_and_w)
      new_bits = sum(total_bits) / len(total_bits) if len(total_bits) else 0.

    else:
      aggregated_params = tree_util.tree_mean(quantized_p_and_w)
      total_num_params = tree_util.tree_size(aggregated_params)
      total_num_floats = 2 * num_leaves(aggregated_params)
      # 32 bits for every float and log2(num_levels) bit for every parameter.
      new_bits = math.log2(
          num_levels) * total_num_params + 32 * total_num_floats
    new_state = CompressionState(aggregator_state.num_bits + new_bits, rng)  # pyrefly: ignore[bad-argument-count]
    return aggregated_params, new_state

  return aggregator.Aggregator(init, apply)  # pyrefly: ignore[bad-argument-count]


def rotated_uniform_stochastic_quantizer(num_levels: int,
                                         rng: PRNGKey) -> aggregator.Aggregator:
  """Returns (weighted) mean of input uniformly quantized trees with rotation

  using the algorithm in https://arxiv.org/pdf/1611.00429.pdf.

  Args:
    num_levels: number of levels of quantization.
    rng: PRNGKey used for compression.

  Returns:
    Compression aggregator.
  """

  def init():
    return CompressionState(0.0, rng)  # pyrefly: ignore[bad-argument-count]

  def apply(
      clients_params_and_weights: Iterable[Tuple[ClientId, Params, float]],
      aggregator_state: CompressionState) -> Tuple[Params, CompressionState]:

    rng, rotation_rng = jax.random.split(aggregator_state.rng)

    def quantize_params_and_weight(client_params_and_weight, rng):
      _, params, weight = client_params_and_weight
      params, shapes = walsh_hadamard.structured_rotation_pytree(
          params, rotation_rng)
      params = walsh_hadamard.inverse_structured_rotation_pytree(
          uniform_stochastic_quantize_pytree(params, num_levels, rng),
          rotation_rng, shapes)
      return params, weight

    rng, use_rng = jax.random.split(rng)
    # TODO(theertha): remove the usage of hk.PRNGSequence.
    rng_seq = hk.PRNGSequence(use_rng)
    clients_params_and_weight_rng = zip(clients_params_and_weights, rng_seq)
    quantized_p_and_w = itertools.starmap(quantize_params_and_weight,
                                          clients_params_and_weight_rng)
    aggregated_params = tree_util.tree_mean(quantized_p_and_w)
    total_num_params = tree_util.tree_size(aggregated_params)
    total_num_floats = 2 * num_leaves(aggregated_params)
    # 32 bits for every float used and log2(num_levels) bit for every parameter.
    new_bits = math.log2(num_levels) * total_num_params + 32 * total_num_floats
    new_state = CompressionState(aggregator_state.num_bits + new_bits, rng)  # pyrefly: ignore[bad-argument-count]
    return aggregated_params, new_state

  return aggregator.Aggregator(init, apply)  # pyrefly: ignore[bad-argument-count]


@jax.jit
def drive_pytree(params: Params) -> Params:
  """Runs DRIVE quantization on a given pytree."""
  leaves, tree_def = jax.tree_util.tree_flatten(params)
  new_leaves = []
  for leaf in leaves:
    # this uses the unbiased scale from section 4.2 in DRIVE's paper (Scale = norm2(R(x))**2 / norm1(R(x)) )
    new_leaves.append(jnp.sum(jnp.power(leaf, 2)) * jnp.sign(leaf) / jnp.sum(jnp.abs(leaf)))
  return jax.tree_util.tree_unflatten(tree_def, new_leaves)


def structured_drive_quantizer(rng: PRNGKey) -> aggregator.Aggregator:
  """Returns (weighted) mean using the structured DRIVE algorithm.

  Paper: https://arxiv.org/pdf/2105.08339.pdf.

  Args:
    rng: PRNGKey used for compression.

  Returns:
    Compression aggregator.
  """

  def init():
    return CompressionState(0.0, rng)  # pyrefly: ignore[bad-argument-count]

  def apply(
      clients_params_and_weights: Iterable[Tuple[ClientId, Params, float]],
      aggregator_state: CompressionState) -> Tuple[Params, CompressionState]:

    rng, rotation_rng = jax.random.split(aggregator_state.rng)
    rotation_rng_seq = hk.PRNGSequence(rotation_rng)
    clients_params_and_weight_rng = zip(clients_params_and_weights,
                                        rotation_rng_seq)
    def quantize_params_and_weight(client_params_and_weight, client_rng):
      _, params, weight = client_params_and_weight
      rotated_param, shapes = walsh_hadamard.structured_rotation_pytree(
          params, client_rng)
      return walsh_hadamard.inverse_structured_rotation_pytree(
          drive_pytree(rotated_param), client_rng, shapes), weight

    quantized_p_and_w = itertools.starmap(quantize_params_and_weight,
                                          clients_params_and_weight_rng)

    aggregated_params = tree_util.tree_mean(quantized_p_and_w)

    total_num_params = tree_util.tree_size(aggregated_params)
    total_num_floats = 2 * num_leaves(aggregated_params)
    # 32 bits for every float used and one bit for every parameter.
    new_bits = total_num_params + 32 * total_num_floats
    new_state = CompressionState(aggregator_state.num_bits + new_bits, rng)  # pyrefly: ignore[bad-argument-count]
    return aggregated_params, new_state

  return aggregator.Aggregator(init, apply)  # pyrefly: ignore[bad-argument-count]


def terngrad_quantize(v: jnp.ndarray, rng: PRNGKey) -> jnp.ndarray:
  """Terngrad algorithm https://arxiv.org/abs/1705.07878.

  Args:
    v: vector to be quantized.
    rng: jax random key.

  Returns:
    Quantized array.
  """
  sigma = jnp.std(v)
  v = jnp.where(jnp.abs(v) > 2.5 * sigma, 2.5 * sigma * jnp.sign(v), v)
  return binary_stochastic_quantize(jnp.abs(v), rng, 0., jnp.amax(  # pyrefly: ignore[bad-argument-type]
      jnp.abs(v))) * jnp.sign(v)


@jax.jit
def terngrad_quantize_pytree(params: Params, rng: PRNGKey) -> Params:
  """Applies terngrad_quantize for all leafs of the pytree.

  Args:
    params: pytree to be quantized.
    rng: jax random key.

  Returns:
    Quantized pytree.
  """
  leaves, tree_def = jax.tree_util.tree_flatten(params)
  rngs = jax.random.split(rng, len(leaves))
  new_leaves = []
  for l, r in zip(leaves, rngs):
    new_leaves.append(terngrad_quantize(l, r))
  return jax.tree_util.tree_unflatten(tree_def, new_leaves)


def terngrad_quantizer(rng: PRNGKey) -> aggregator.Aggregator:
  """Returns (weighted) mean of terngrad algorithm.

  Paper: https://arxiv.org/abs/1705.07878.

  Args:
    rng: PRNGKey used for compression.

  Returns:
    Compression aggregator.
  """

  def init():
    return CompressionState(0.0, rng)  # pyrefly: ignore[bad-argument-count]

  def apply(
      clients_params_and_weights: Iterable[Tuple[ClientId, Params, float]],
      aggregator_state: CompressionState) -> Tuple[Params, CompressionState]:

    def quantize_params_and_weight(client_params_and_weight, rng):
      _, params, weight = client_params_and_weight
      return terngrad_quantize_pytree(params, rng), weight

    rng, use_rng = jax.random.split(aggregator_state.rng)
    # TODO(theertha): remove the usage of hk.PRNGSequence.
    rng_seq = hk.PRNGSequence(use_rng)
    clients_params_and_weight_rng = zip(clients_params_and_weights, rng_seq)
    quantized_p_and_w = itertools.starmap(quantize_params_and_weight,
                                          clients_params_and_weight_rng)
    aggregated_params = tree_util.tree_mean(quantized_p_and_w)
    total_num_params = tree_util.tree_size(aggregated_params)
    total_num_floats = 2 * num_leaves(aggregated_params)
    # 32 bits for every float used and log2(3) bit for every parameter.
    new_bits = math.log2(3) * total_num_params + 32 * total_num_floats
    new_state = CompressionState(aggregator_state.num_bits + new_bits, rng)  # pyrefly: ignore[bad-argument-count]
    return aggregated_params, new_state

  return aggregator.Aggregator(init, apply)  # pyrefly: ignore[bad-argument-count]


@partial(jax.jit, static_argnames=('nbits', 'block_size'))
def block_uniform_quantize(v: jnp.ndarray,
                           nbits: int = 4,
                           block_size: int = 64,
                           rng: Optional[PRNGKey] = None) -> jnp.ndarray:
  """Block-wise uniform quantization algorithm with dynamic per-block scaling.

  Args:
    v: Vector/array to be quantized.
    nbits: Number of bits for quantization. Defaults to 4.
    block_size: Number of elements per quantization block. Defaults to 64.
    rng: Optional PRNGKey for unbiased stochastic rounding. If None, uses
      deterministic nearest-integer rounding.

  Returns:
    Quantized array of the same shape and dtype as v.
  """
  orig_shape = v.shape
  flat = v.reshape(-1)
  n = flat.size
  if n == 0:
    return v

  pad_len = (block_size - (n % block_size)) % block_size
  padded = jnp.pad(flat, (0, pad_len))
  blocks = padded.reshape(-1, block_size)
  max_abs = jnp.max(jnp.abs(blocks), axis=-1, keepdims=True)
  q_max = (1 << (nbits - 1)) - 1
  q_min = -(1 << (nbits - 1))
  scale = max_abs / q_max
  safe_scale = jnp.where(max_abs == 0.0, 1.0, scale)
  norm = blocks / safe_scale

  if rng is not None:
    rand = jax.random.uniform(key=rng, shape=norm.shape, minval=-0.5, maxval=0.5)
    norm = norm + rand

  q = jnp.where(max_abs == 0.0, 0.0, jnp.clip(jnp.round(norm), q_min, q_max))
  restored = q * scale
  return restored.reshape(-1)[:n].reshape(orig_shape).astype(v.dtype)


def block_uniform_quantize_pytree(params: Params,
                                  nbits: int = 4,
                                  block_size: int = 64,
                                  rng: Optional[PRNGKey] = None) -> Params:
  """Applies block_uniform_quantize for all leaves of the pytree.

  Args:
    params: Pytree to be quantized.
    nbits: Number of quantization bits.
    block_size: Number of elements per block.
    rng: Optional JAX random key for stochastic rounding.

  Returns:
    Quantized pytree.
  """
  leaves, tree_def = jax.tree_util.tree_flatten(params)
  if rng is not None:
    rngs = jax.random.split(rng, len(leaves))
  else:
    rngs = [None] * len(leaves)
  new_leaves = [
      block_uniform_quantize(l, nbits=nbits, block_size=block_size, rng=r)
      for l, r in zip(leaves, rngs)
  ]
  return jax.tree_util.tree_unflatten(tree_def, new_leaves)


def block_quantizer(
    nbits: int = 4,
    block_size: int = 64,
    stochastic: bool = False,
    rng: Optional[PRNGKey] = None) -> aggregator.Aggregator:
  """Returns (weighted) mean of input block-wise quantized trees.

  Args:
    nbits: Number of quantization bits (e.g. 4 or 8). Defaults to 4.
    block_size: Number of elements per block for dynamic scaling. Defaults to 64.
    stochastic: Whether to apply stochastic rounding for unbiased aggregation.
    rng: Optional PRNGKey used for compression state.

  Returns:
    Compression aggregator.
  """
  if not isinstance(nbits, int) or isinstance(nbits, bool):
    raise TypeError(f"nbits must be an integer, got {type(nbits)}")
  if nbits < 2 or nbits > 16:
    raise ValueError(f"nbits must be between 2 and 16, got {nbits}")
  if not isinstance(block_size, int) or isinstance(block_size, bool):
    raise TypeError(f"block_size must be an integer, got {type(block_size)}")
  if block_size <= 0:
    raise ValueError(f"block_size must be a positive integer, got {block_size}")
  if not isinstance(stochastic, bool):
    raise TypeError(f"stochastic must be a boolean, got {type(stochastic)}")

  if rng is None:
    rng = jax.random.PRNGKey(0)

  def init():
    return CompressionState(0.0, rng)  # pyrefly: ignore[bad-argument-count]

  def apply(
      clients_params_and_weights: Iterable[Tuple[ClientId, Params, float]],
      aggregator_state: CompressionState) -> Tuple[Params, CompressionState]:

    rng, use_rng = jax.random.split(aggregator_state.rng)
    rng_seq = hk.PRNGSequence(use_rng)

    def quantize_client(client_params_and_weight, client_rng):
      _, params, weight = client_params_and_weight
      q_rng = client_rng if stochastic else None
      q_params = block_uniform_quantize_pytree(
          params, nbits=nbits, block_size=block_size, rng=q_rng)
      return q_params, weight

    clients_params_and_weight_rng = zip(clients_params_and_weights, rng_seq)
    quantized_p_and_w = itertools.starmap(quantize_client,
                                          clients_params_and_weight_rng)

    aggregated_params = tree_util.tree_mean(quantized_p_and_w)
    leaves = jax.tree_util.tree_leaves(aggregated_params)
    total_num_params = sum(l.size for l in leaves)
    total_num_blocks = sum(math.ceil(l.size / block_size) for l in leaves)

    # nbits per parameter + 32 bits for each float32 block scale factor
    new_bits = (nbits * total_num_params) + (32 * total_num_blocks)
    new_state = CompressionState(aggregator_state.num_bits + new_bits, rng)  # pyrefly: ignore[bad-argument-count]
    return aggregated_params, new_state

  return aggregator.Aggregator(init, apply)  # pyrefly: ignore[bad-argument-count]
