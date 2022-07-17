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
    self.assertEqual(new_state.num_bits, 78.08298)
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

if __name__ == '__main__':
  absltest.main()
