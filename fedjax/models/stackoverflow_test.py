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
"""Tests for fedjax.models.stackoverflow."""

from typing import Hashable

from absl.testing import absltest

from fedjax.core import tree_util
from fedjax.models import stackoverflow

import jax
import jax.numpy as jnp
import numpy.testing as npt


class StackOverflowModelTest(absltest.TestCase):

  def check_model(self, model):
    params = model.init(jax.random.PRNGKey(0))
    batch = {
        'x': jnp.ones((5, 3), dtype=jnp.int32),
        'y': jnp.ones((5, 3), dtype=jnp.int32)
    }
    with self.subTest('apply_for_train'):
      preds = model.apply_for_train(params, batch)
      self.assertTupleEqual(preds.shape, (5, 3, 10000 + 4))
    with self.subTest('apply_for_eval'):
      preds = model.apply_for_eval(params, batch)
      self.assertTupleEqual(preds.shape, (5, 3, 10000 + 4))
    with self.subTest('train_loss'):
      preds = model.apply_for_train(params, batch)
      train_loss = model.train_loss(batch, preds)
      self.assertTupleEqual(train_loss.shape, (5,))
    with self.subTest('hashable'):
      self.assertIsInstance(model, Hashable)

  def test_create_lstm_model(self):
    model = stackoverflow.create_lstm_model()
    params = model.init(jax.random.PRNGKey(0))
    self.assertEqual(tree_util.tree_size(params), 4050748)
    self.check_model(model)

  def test_create_lstm_model_share_embeddings(self):
    model = stackoverflow.create_lstm_model(share_input_output_embeddings=True)
    params = model.init(jax.random.PRNGKey(0))
    self.assertEqual(tree_util.tree_size(params), 3090364)
    self.check_model(model)

  def test_expected_length_scale_loss(self):
    batch = {'y': jnp.ones((5, 3), dtype=jnp.int32)}
    preds = jnp.ones((5, 3, 10000))
    expected_length = 3.
    model = stackoverflow.create_lstm_model()
    scaled_model = stackoverflow.create_lstm_model(
        expected_length=expected_length)
    train_loss = model.train_loss(batch, preds)
    scaled_train_loss = scaled_model.train_loss(batch, preds)
    npt.assert_array_equal(train_loss * (1 / expected_length),
                           scaled_train_loss)


if __name__ == '__main__':
  absltest.main()
