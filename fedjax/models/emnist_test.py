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
"""Tests for fedjax.models.emnist."""

from absl.testing import absltest

from fedjax.core import tree_util
from fedjax.models import emnist

import jax
import jax.numpy as jnp


class EmnistModelTest(absltest.TestCase):

  def check_model(self, model):
    rng = jax.random.PRNGKey(0)
    params = model.init(rng)
    batch = {'x': jnp.ones((5, 28, 28, 1)), 'y': jnp.ones((5,))}
    with self.subTest('apply_for_train'):
      preds = model.apply_for_train(params, batch, rng)
      self.assertTupleEqual(preds.shape, (5, 62))
    with self.subTest('apply_for_eval'):
      preds = model.apply_for_eval(params, batch)
      self.assertTupleEqual(preds.shape, (5, 62))
    with self.subTest('train_loss'):
      preds = model.apply_for_train(params, batch, rng)
      train_loss = model.train_loss(batch, preds)
      self.assertTupleEqual(train_loss.shape, (5,))

  def test_create_conv_model(self):
    model = emnist.create_conv_model(only_digits=False)
    params = model.init(jax.random.PRNGKey(0))
    self.assertEqual(tree_util.tree_size(params), 1206590)
    self.check_model(model)

  def test_create_dense_model(self):
    model = emnist.create_dense_model(only_digits=False, hidden_units=200)
    params = model.init(jax.random.PRNGKey(0))
    self.assertEqual(tree_util.tree_size(params), 209662)
    self.check_model(model)

  def test_create_logistic_model(self):
    model = emnist.create_logistic_model(only_digits=False)
    params = model.init(jax.random.PRNGKey(0))
    self.assertEqual(tree_util.tree_size(params), 48670)
    self.check_model(model)

  def test_create_stax_dense_model(self):
    model = emnist.create_stax_dense_model(only_digits=False, hidden_units=200)
    params = model.init(jax.random.PRNGKey(0))
    self.assertEqual(tree_util.tree_size(params), 209662)
    self.check_model(model)

if __name__ == '__main__':
  absltest.main()
