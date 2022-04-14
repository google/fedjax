# Copyright 2022 Google LLC
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
"""Tests for fedjax.models.cifar100."""

from absl.testing import absltest

from fedjax.core import tree_util
from fedjax.models import cifar100

import jax
import jax.numpy as jnp


class Cifar100ModelTest(absltest.TestCase):

  def check_model(self, model):
    rng = jax.random.PRNGKey(0)
    params = model.init(rng)
    batch_size = 5
    num_classes = 100
    batch = {
        'x': jnp.ones((batch_size, 24, 24, 3)),
        'y': jnp.ones((batch_size,)),
    }
    with self.subTest('apply_for_train'):
      preds = model.apply_for_train(params, batch, rng)
      self.assertTupleEqual(preds.shape, (batch_size, num_classes))
    with self.subTest('apply_for_eval'):
      preds = model.apply_for_eval(params, batch)
      self.assertTupleEqual(preds.shape, (batch_size, num_classes))
    with self.subTest('train_loss'):
      preds = model.apply_for_train(params, batch, rng)
      train_loss = model.train_loss(batch, preds)
      self.assertTupleEqual(train_loss.shape, (batch_size,))

  def test_create_logistic_model(self):
    model = cifar100.create_logistic_model()
    params = model.init(jax.random.PRNGKey(0))
    self.assertEqual(tree_util.tree_size(params), 172900)
    self.check_model(model)


if __name__ == '__main__':
  absltest.main()
