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
"""Tests for fedjax.aggregators.aggregator."""
from absl.testing import absltest

from fedjax.aggregators import aggregator
import jax.numpy as jnp
import numpy.testing as npt


class AggregatorTest(absltest.TestCase):

  def test_mean_aggregator(self):
    delta_params_and_weights = [('a', {
        'w': jnp.array([1., 2., 3.])
    }, 2.), ('b', {
        'w': jnp.array([2., 4., 6.])
    }, 4.), ('c', {
        'w': jnp.array([1., 3., 5.])
    }, 2.)]
    mean_aggregator = aggregator.mean_aggregator()

    aggregator_state = mean_aggregator.init()
    mean_params, _ = mean_aggregator.apply(delta_params_and_weights,
                                           aggregator_state)

    npt.assert_array_equal(mean_params['w'], [1.5, 3.25, 5.])


if __name__ == '__main__':
  absltest.main()
