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
    v_min = jnp.amin(v)
  if v_max is None:
    v_max = jnp.amax(v)
  v = jnp.nan_to_num((v - v_min) / (v_max - v_min))
  v = jnp.maximum(0., jnp.minimum(v, 1.))
  rand = jax.random.uniform(key=rng, shape=v.shape)
  return jnp.where(rand > v, v_min, v_max)


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
    v_min = jnp.amin(v)
  if v_max is None:
    v_max = jnp.amax(v)
  v = jnp.nan_to_num((v - v_min) / (v_max - v_min))
  v = jnp.maximum(0., jnp.minimum(v, 1.))
  # Compute the upper and lower boundary of each value.
  v_ceil = jnp.ceil(v * (num_levels - 1)) / (num_levels - 1)
  v_floor = jnp.floor(v * (num_levels - 1)) / (num_levels - 1)
  # uniformly quantize between v_ceil and v_floor.
  rand = jax.random.uniform(key=rng, shape=v.shape)
  threshold = jnp.nan_to_num((v - v_floor) / (v_ceil - v_floor))
  quantized = jnp.where(rand > threshold, v_floor, v_ceil)
  # Rescale the values and return it.
  return v_min + quantized * (v_max - v_min)


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


def arithmetic_encoding_num_bits(v: jnp.ndarray) -> int:
  """Computes number of bits needed to store v via arithmetic coding."""
  v = jnp.nan_to_num(v)
  v = v.flatten()
  uniq = jnp.unique(v)
  entropy = _entropy(v, uniq)
  return v.size * entropy + 2 * 32 + 2


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
    return CompressionState(0.0, rng)

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
    new_state = CompressionState(aggregator_state.num_bits + new_bits, rng)
    return aggregated_params, new_state

  return aggregator.Aggregator(init, apply)


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
    return CompressionState(0.0, rng)

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
    new_state = CompressionState(aggregator_state.num_bits + new_bits, rng)
    return aggregated_params, new_state

  return aggregator.Aggregator(init, apply)


@jax.jit
def drive_pytree(params: Params) -> Params:
  """Runs DRIVE quantization on a given pytree."""
  leaves, tree_def = jax.tree_util.tree_flatten(params)
  new_leaves = []
  for leaf in leaves:
    new_leaves.append(jnp.sum(jnp.abs(leaf)) * jnp.sign(leaf) / leaf.size)
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
    return CompressionState(0.0, rng)

  def apply(
      clients_params_and_weights: Iterable[Tuple[ClientId, Params, float]],
      aggregator_state: CompressionState) -> Tuple[Params, CompressionState]:

    rng, rotation_rng = jax.random.split(aggregator_state.rng)

    def quantize_params_and_weight(client_id, params, weight):
      del client_id
      rotated_param, shapes = walsh_hadamard.structured_rotation_pytree(
          params, rotation_rng)
      return walsh_hadamard.inverse_structured_rotation_pytree(
          drive_pytree(rotated_param), rotation_rng, shapes), weight

    quantized_p_and_w = itertools.starmap(quantize_params_and_weight,
                                          clients_params_and_weights)

    aggregated_params = tree_util.tree_mean(quantized_p_and_w)

    total_num_params = tree_util.tree_size(aggregated_params)
    total_num_floats = 2 * num_leaves(aggregated_params)
    # 32 bits for every float used and one bit for every parameter.
    new_bits = total_num_params + 32 * total_num_floats
    new_state = CompressionState(aggregator_state.num_bits + new_bits, rng)
    return aggregated_params, new_state

  return aggregator.Aggregator(init, apply)


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
  return binary_stochastic_quantize(jnp.abs(v), rng, 0., jnp.amax(
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
    return CompressionState(0.0, rng)

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
    new_state = CompressionState(aggregator_state.num_bits + new_bits, rng)
    return aggregated_params, new_state

  return aggregator.Aggregator(init, apply)
