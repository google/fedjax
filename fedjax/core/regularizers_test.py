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
"""Tests for fedjax.core.regularizers."""

from fedjax.core import regularizers
from fedjax.core import test_util
import jax
import tensorflow as tf


class RegularizerTest(tf.test.TestCase):

  def test_l2_regularizer(self):
    params = test_util.create_mock_state(seed=0).params
    output = regularizers.L2Regularizer()(params)
    self.assertAlmostEqual(output, 37.64189)

  def test_l2_regularizer_weight(self):
    params = test_util.create_mock_state(seed=0).params
    original_output = regularizers.L2Regularizer()(params)
    output = regularizers.L2Regularizer(weight=0.2)(params)
    self.assertAlmostEqual(output, original_output * 0.2)

  def test_l2_regularizer_parameter_weight(self):
    params = test_util.create_mock_state(seed=0).params
    original_output = regularizers.L2Regularizer()(params)

    param_weights = jax.tree_map(lambda leaf: 2 * jax.numpy.ones(leaf.shape),
                                 params)
    output = regularizers.L2Regularizer(
        weight=1.0, param_weights=param_weights)(
            params)
    self.assertAlmostEqual(output, 2 * original_output, delta=1e-5)

  def test_l2_regularizer_evaluation_with_center(self):
    params = test_util.create_mock_state(seed=0).params
    original_output = regularizers.L2Regularizer()(params)
    center_params = jax.tree_map(lambda l: l * 0.2, params)
    output = regularizers.L2Regularizer(center_params=center_params)(params)
    self.assertAlmostEqual(output, original_output * (1. - 0.2)**2)


if __name__ == '__main__':
  tf.test.main()
