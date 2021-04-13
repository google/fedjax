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

from typing import Iterable, NamedTuple, Tuple, TypeVar

from fedjax import core
from fedjax.experimental.aggregators import aggregator
import jax
import jax.numpy as jnp

W = TypeVar('W')


class CompressionState(NamedTuple):
  """State of default aggregator passed between rounds.

  Attributes:
    total_weight: the number of samples.
    num_bits: number of bits transmitted.
  """
  total_weight: float
  num_bits: float


def binary_stochastic_quantize(v, rng):
  v_min = jnp.amin(v)
  v_max = jnp.amax(v)
  v = (v - v_min) / (v_max - v_min + 1e-15)
  rand = jax.random.uniform(key=rng, shape=v.shape)
  return jnp.where(rand > v, v_min, v_max)


def num_leaves(pytree):
  return len(jax.tree_util.tree_leaves(pytree))


def num_params(pytree):
  x = jax.tree_util.tree_leaves(pytree)
  return sum(i.size for i in x)


class BinaryStochasticQuantizer(aggregator.Aggregator):
  """Binary stochastic algorithm in https://arxiv.org/pdf/1611.00429.pdf."""

  def init_state(self):
    return CompressionState(total_weight=0.0, num_bits=0.0)

  def aggregate(self, aggregator_state: CompressionState,
                params_and_weight: Iterable[Tuple[W, float]],
                rng_seq: core.PRNGSequence) -> Tuple[CompressionState, W]:
    """Returns (weighted) mean of input trees and weights.

    Args:
      aggregator_state: state of the input.
      params_and_weight: Iterable of tuples of pytrees and associated weights.
      rng_seq: Random sequence.

    Returns:
      New state, (Weighted) mean of input trees and weights.
    """

    def quantize_params_and_weight(param_weight_rng):
      param_weight, rng = param_weight_rng
      param, weight = param_weight

      def binary_stochastic_quantize_with_rng(param):
        return binary_stochastic_quantize(param, rng)
      return jax.tree_map(binary_stochastic_quantize_with_rng, param), weight

    # TODO(theertha): avoid the need to copying the entire sequence to memory.

    params_and_weight_rng = zip(params_and_weight, rng_seq)
    quantized_p_and_w = map(quantize_params_and_weight,
                            params_and_weight_rng)
    quantized_p_and_w = list(quantized_p_and_w)
    weights = [weight for params, weight in quantized_p_and_w]
    new_weight = sum(weights)
    params = [params for params, weight in quantized_p_and_w]
    total_num_floats = sum([num_leaves(param) for param in params])
    total_num_params = sum([num_params(param) for param in params])
    # 32 bits for every float used and one bit for every parameter.
    new_bits = total_num_params + 64 * total_num_floats
    new_state = CompressionState(
        total_weight=aggregator_state.total_weight + new_weight,
        num_bits=aggregator_state.num_bits + new_bits)
    return core.tree_mean(quantized_p_and_w), new_state
