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
"""Tests for fedjax.aggregators.compression."""
from absl.testing import absltest

from fedjax.aggregators import compression
import jax
import jax.numpy as jnp
import numpy.testing as npt

jax.config.update('jax_threefry_partitionable', False)


class CompressionTest(absltest.TestCase):

  def test_num_leaves(self):
    params = {'w': jnp.array([1., 2., 3.]), 'b': jnp.array([1.])}
    self.assertEqual(compression.num_leaves(params), 2)

  def test_binary_stochastic_quantize_identity(self):
    # If the vector has only two distinct values, it should not change.
    v = jnp.array([0., 2., 2.])
    rng = jax.random.PRNGKey(42)
    compressed_v = compression.binary_stochastic_quantize(v, rng)
    npt.assert_array_equal(compressed_v, v)

  def test_binary_stochastic_quantize_unbiasedness(self):
    v = jnp.array([0., 1., 2.])
    rng = jax.random.PRNGKey(42)
    compressed_sum = jnp.zeros_like(v)
    for _ in range(500):
      rng, use_rng = jax.random.split(rng)
      compressed_sum += compression.binary_stochastic_quantize(v, use_rng)
    npt.assert_array_almost_equal(compressed_sum / 500, v, decimal=2)

  def test_uniform_stochastic_quantize_identity(self):
    # If the vector has the same number of distinct values as the number of
    # quantization levels, it should not change.
    v = jnp.array([0., 2., 2., 4.])
    rng = jax.random.PRNGKey(42)
    compressed_v = compression.uniform_stochastic_quantize(v, 3, rng)
    npt.assert_array_equal(compressed_v, v)

  def test_uniform_stochastic_quantize_all_equal(self):
    v = jnp.array([4., 4., 4., 4.])
    rng = jax.random.PRNGKey(42)
    compressed_v = compression.uniform_stochastic_quantize(v, 4, rng)
    npt.assert_array_equal(compressed_v, v)

  def test_uniform_stochastic_quantize_unbiasedness_one_dim(self):
    v = jnp.array([0., 1., 100.])
    rng = jax.random.PRNGKey(42)
    compressed_sum = jnp.zeros_like(v)
    for _ in range(500):
      rng, use_rng = jax.random.split(rng)
      compressed_sum += compression.uniform_stochastic_quantize(v, 125, use_rng)
    npt.assert_array_almost_equal(compressed_sum / 500, v, decimal=2)

  def test_uniform_stochastic_quantize_unbiasedness_two_dim(self):
    v = jnp.array([[0., 1., 100.], [0.3, 2.3, 45.]])
    rng = jax.random.PRNGKey(42)
    compressed_sum = jnp.zeros_like(v)
    for _ in range(500):
      rng, use_rng = jax.random.split(rng)
      compressed_sum += compression.uniform_stochastic_quantize(v, 125, use_rng)
    npt.assert_array_almost_equal(compressed_sum / 500, v, decimal=1)

  def test_arithmetic_encoding_num_bits(self):
    v = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    npt.assert_array_almost_equal(
        compression.arithmetic_encoding_num_bits(v), [89.82311], decimal=3)

  def test_uniform_stochastic_quantizer(self):
    delta_params_and_weights = [('a', {
        'w': jnp.array([1., 2., 3.])
    }, 2.), ('b', {
        'w': jnp.array([2., 4., 6.])
    }, 4.), ('c', {
        'w': jnp.array([1., 3., 5.])
    }, 2.)]

    quantizer = compression.uniform_stochastic_quantizer(
        3, jax.random.PRNGKey(0))
    init_aggregator_state = quantizer.init()
    quantized_params, new_state = quantizer.apply(delta_params_and_weights,
                                                  init_aggregator_state)
    self.assertEqual(new_state.num_bits, 68.75489)
    npt.assert_array_equal(quantized_params['w'], [1.5, 3.25, 5.])

  def test_uniform_stochastic_quantizer_arithmetic_coding(self):
    delta_params_and_weights = [('a', {
        'w': jnp.array([1., 2., 3.])
    }, 2.), ('b', {
        'w': jnp.array([2., 4., 6.])
    }, 4.), ('c', {
        'w': jnp.array([1., 3., 5.])
    }, 2.)]

    quantizer = compression.uniform_stochastic_quantizer(
        3, jax.random.PRNGKey(0), 'arithmetic')
    init_aggregator_state = quantizer.init()
    quantized_params, new_state = quantizer.apply(delta_params_and_weights,
                                                  init_aggregator_state)
    npt.assert_almost_equal(new_state.num_bits, 78.08298, decimal=4)
    npt.assert_array_equal(quantized_params['w'], [1.5, 3.25, 5.])

  def test_rotated_uniform_stochastic_quantizer(self):
    delta_params_and_weights = [('a', {
        'w': jnp.array([1., 2., 3.])
    }, 2.), ('b', {
        'w': jnp.array([2., 4., 6.])
    }, 4.), ('c', {
        'w': jnp.array([1., 3., 5.])
    }, 2.)]

    quantizer = compression.rotated_uniform_stochastic_quantizer(
        2, jax.random.PRNGKey(0))
    state = quantizer.init()
    all_params = []
    for _ in range(2000):
      quantized_params, state = quantizer.apply(delta_params_and_weights, state)
      all_params.append(quantized_params)
    self.assertEqual(state.num_bits, 67 * 2000)
    params_mean = jnp.mean(
        jnp.array([params['w'] for params in all_params]), axis=0)
    npt.assert_array_almost_equal(params_mean, [1.5, 3.25, 5.], decimal=1)

  def test_structured_drive_pytree(self):
    x = {'w': jnp.array([1., -2., 3.])}
    y = compression.drive_pytree(x)
    npt.assert_array_almost_equal(
        y['w'], jnp.array([2.333333, -2.333333,  2.333333]), decimal=4)

  def test_structured_drive_quantizer(self):
    delta_params_and_weights = [('a', {
        'w': jnp.array([1., 2., 3.])
    }, 2.), ('b', {
        'w': jnp.array([2., 4., 6.])
    }, 4.), ('c', {
        'w': jnp.array([1., 3., 5.])
    }, 2.)]

    quantizer = compression.structured_drive_quantizer(jax.random.PRNGKey(0))
    aggregator_state = quantizer.init()
    aggregated_params_list = []
    for _ in range(100):
      aggregated_params, aggregator_state = quantizer.apply(
          delta_params_and_weights, aggregator_state)
      aggregated_params_list.append(aggregated_params)
    self.assertEqual(aggregator_state.num_bits, 67 * 100)
    mean_aggregated_params = sum([
        aggregated_params['w'] for aggregated_params in aggregated_params_list
    ]) / 100
    npt.assert_array_almost_equal(
        mean_aggregated_params, [1.458334, 1.458334, 6.125], decimal=4)

  def test_terngrad_quantize_identity(self):
    # If the vector has only two distinct values and sigma > 2.5 * v_max,
    # it should not change.
    v = jnp.array([0., 2., 2.])
    rng = jax.random.PRNGKey(42)
    compressed_v = compression.terngrad_quantize(v, rng)
    npt.assert_array_equal(compressed_v, v)

  def test_terngrad_quantize_clipping(self):
    # If the vector has only two distinct values and sigma > 2.5 * v_max,
    # it should not change.
    v = jnp.zeros(100)
    v = v.at[0].set(100)
    v = v.at[1].set(-100)
    rng = jax.random.PRNGKey(42)
    compressed_v = compression.terngrad_quantize(v, rng)
    expected_v = v
    expected_v = expected_v.at[0].set(35.355339)
    expected_v = expected_v.at[1].set(-35.355339)
    npt.assert_array_equal(compressed_v, expected_v)

  def test_terngrad_quantizer(self):
    delta_params_and_weights = [('a', {
        'w': jnp.array([1., 2., 3.])
    }, 2.), ('b', {
        'w': jnp.array([2., 4., 6.])
    }, 4.), ('c', {
        'w': jnp.array([1., 3., 5.])
    }, 2.)]

    quantizer = compression.terngrad_quantizer(jax.random.PRNGKey(0))
    init_aggregator_state = quantizer.init()
    quantized_params, new_state = quantizer.apply(delta_params_and_weights,
                                                  init_aggregator_state)
    self.assertEqual(new_state.num_bits, 68.75489)
    npt.assert_array_almost_equal(
        quantized_params['w'], [0.51031, 2.551552, 3.572173], decimal=4)

  def test_block_uniform_quantize_invalid_parameters(self):
    with self.assertRaises(ValueError):
      compression.block_quantizer(nbits=1)
    with self.assertRaises(ValueError):
      compression.block_quantizer(nbits=17)
    with self.assertRaises((ValueError, TypeError)):
      compression.block_quantizer(block_size=0)
    with self.assertRaises((ValueError, TypeError)):
      compression.block_quantizer(block_size=-4)
    with self.assertRaises((ValueError, TypeError)):
      compression.block_quantizer(stochastic='no')

  def test_block_uniform_quantize_identity(self):
    # Distinct values aligned with scale factor max / 7 should not change.
    v = jnp.array([0.0, 7.0, -7.0, 3.0])
    quantized_v = compression.block_uniform_quantize(
        v, nbits=4, block_size=4, rng=None)
    npt.assert_array_almost_equal(quantized_v, v, decimal=5)

  def test_block_uniform_quantize_all_equal(self):
    v = jnp.array([3.0, 3.0, 3.0, 3.0])
    quantized_v = compression.block_uniform_quantize(
        v, nbits=4, block_size=4, rng=None)
    npt.assert_array_almost_equal(quantized_v, v, decimal=5)

  def test_block_uniform_quantize_zeros(self):
    zeros = jnp.zeros((8, 8))
    quantized_zeros = compression.block_uniform_quantize(
        zeros, nbits=4, block_size=16, rng=None)
    npt.assert_array_equal(quantized_zeros, zeros)

  def test_block_uniform_quantize_unaligned_length(self):
    key = jax.random.PRNGKey(42)
    v = jax.random.normal(key, shape=(37,))
    quantized_v = compression.block_uniform_quantize(
        v, nbits=4, block_size=16, rng=None)
    self.assertEqual(quantized_v.shape, v.shape)
    mse = jnp.mean((v - quantized_v) ** 2)
    self.assertLess(mse, 0.05)

  def test_block_uniform_quantize_stochastic_unbiasedness(self):
    v = jnp.array([0.2, -1.3, 2.7, 0.0])
    rng = jax.random.PRNGKey(42)
    compressed_sum = jnp.zeros_like(v)
    for _ in range(500):
      rng, use_rng = jax.random.split(rng)
      compressed_sum += compression.block_uniform_quantize(
          v, nbits=4, block_size=2, rng=use_rng)
    npt.assert_array_almost_equal(compressed_sum / 500, v, decimal=1)

  def test_block_uniform_quantize_pytree(self):
    pytree = {
        'conv': jnp.ones((4, 4)),
        'bias': jnp.array([0.5, -0.5, 1.0]),
    }
    q_pytree = compression.block_uniform_quantize_pytree(
        pytree, nbits=4, block_size=8)
    self.assertEqual(q_pytree['conv'].shape, pytree['conv'].shape)
    self.assertEqual(q_pytree['bias'].shape, pytree['bias'].shape)
    npt.assert_array_almost_equal(q_pytree['conv'], pytree['conv'], decimal=4)

  def test_block_quantizer_aggregator(self):
    delta_params_and_weights = [
        ('a', {'w': jnp.array([1., 2., 3.])}, 2.),
        ('b', {'w': jnp.array([2., 4., 6.])}, 4.),
    ]
    quantizer = compression.block_quantizer(
        nbits=4, block_size=2, stochastic=False)
    state = quantizer.init()
    agg_params, new_state = quantizer.apply(delta_params_and_weights, state)
    # 3 parameters * 4 bits = 12 bits; 2 blocks * 32 bits = 64 bits -> 76 bits total
    self.assertEqual(new_state.num_bits, 76.0)
    npt.assert_array_almost_equal(
        agg_params['w'], [1.4285716, 3.3333335, 5.0000005], decimal=4)

if __name__ == '__main__':
  absltest.main()
