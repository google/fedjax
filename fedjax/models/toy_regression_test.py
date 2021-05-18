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
"""Tests for fedjax.models.toy_regression."""

from absl.testing import absltest

from fedjax.core import tree_util
from fedjax.models import toy_regression

import jax
import jax.numpy as jnp


class ToyRegressionModelTest(absltest.TestCase):

  def test_create_regression_model(self):
    model = toy_regression.create_regression_model()
    params = model.init(jax.random.PRNGKey(0))
    batch = {'x': jnp.ones((5, 1)), 'y': jnp.ones((5,))}
    self.assertEqual(tree_util.tree_size(params), 1)
    with self.subTest('apply_for_train'):
      preds = model.apply_for_train(params, batch)
      self.assertTupleEqual(preds.shape, ())
    with self.subTest('apply_for_eval'):
      preds = model.apply_for_eval(params, batch)
      self.assertTupleEqual(preds.shape, ())
    with self.subTest('train_loss'):
      preds = model.apply_for_train(params, batch)
      train_loss = model.train_loss(batch, preds)
      self.assertTupleEqual(train_loss.shape, ())


if __name__ == '__main__':
  absltest.main()
