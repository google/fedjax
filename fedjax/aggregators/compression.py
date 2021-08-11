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

from typing import Iterable, Tuple

from fedjax.aggregators import aggregator
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


def binary_stochastic_quantize(v, rng):
  """Binary stochastic algorithm in https://arxiv.org/pdf/1611.00429.pdf.

  Args:
    v: vector to be quantized.
    rng: jax random key.

  Returns:
    Quantized vector.
  """
  v_min = jnp.amin(v)
  v_max = jnp.amax(v)
  v = jnp.nan_to_num((v - v_min) / (v_max - v_min))
  rand = jax.random.uniform(key=rng, shape=v.shape)
  return jnp.where(rand > v, v_min, v_max)


def uniform_stochastic_quantize(v, num_levels, rng):
  """Uniform stochastic algorithm in https://arxiv.org/pdf/1611.00429.pdf.

  Args:
    v: vector to be quantized.
    num_levels: Number of levels of quantization.
    rng: jax random key.

  Returns:
    Quantized vector.
  """
  # Rescale the vector to be between zero to one.
  v_min = jnp.amin(v)
  v_max = jnp.amax(v)
  v = jnp.nan_to_num((v - v_min) / (v_max - v_min))
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
def uniform_stochastic_quantize_pytree(param, num_levels, rng):
  """Applies uniform_stochastic_quantize for all leafs of the pytree.

  Args:
    param: pytree to be quantized.
    num_levels: Number of levels of quantization.
    rng: jax random key.

  Returns:
    Quantized pytree.
  """
  leaves, tree_def = jax.tree_util.tree_flatten(param)
  rngs = jax.random.split(rng, len(leaves))
  new_leaves = []
  for i, r in zip(leaves, rngs):
    new_leaves.append(uniform_stochastic_quantize(i, num_levels, r))
  return jax.tree_util.tree_unflatten(tree_def, new_leaves)


def num_leaves(pytree):
  return len(jax.tree_util.tree_leaves(pytree))


def uniform_stochastic_quantizer(num_levels: int,
                                 rng: PRNGKey) -> aggregator.Aggregator:
  """Returns (weighted) mean of input uniformly quantized trees using the
  uniform stochastic algorithm in https://arxiv.org/pdf/1611.00429.pdf.

  Args:
    num_levels: number of levels of quantization.
    rng: PRNGKey used for compression.

  Returns:
    Compression aggreagtor.
  """

  def init():
    return CompressionState(0.0, rng)

  def apply(
      clients_params_and_weights: Iterable[Tuple[ClientId, Params, float]],
      aggregator_state: CompressionState) -> Tuple[Params, CompressionState]:

    def quantize_params_and_weight(clients_params_and_weight_rng):
      (_, param, weight), rng = clients_params_and_weight_rng

      return uniform_stochastic_quantize_pytree(param, num_levels, rng), weight

    rng, use_rng = jax.random.split(aggregator_state.rng)
    # TODO(theertha): remove the usage of hk.PRNGSequence.
    rng_seq = hk.PRNGSequence(use_rng)
    clients_params_and_weight_rng = zip(clients_params_and_weights, rng_seq)
    quantized_p_and_w = map(quantize_params_and_weight,
                            clients_params_and_weight_rng)
    aggregated_params = tree_util.tree_mean(quantized_p_and_w)
    total_num_params = tree_util.tree_size(aggregated_params)
    total_num_floats = num_leaves(aggregated_params)
    # 32 bits for every float used and one bit for every parameter.
    new_bits = total_num_params + 64 * total_num_floats
    new_state = CompressionState(aggregator_state.num_bits + new_bits, rng)
    return aggregated_params, new_state

  return aggregator.Aggregator(init, apply)
