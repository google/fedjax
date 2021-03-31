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
"""Tests for fedjax.experimental.model."""

from absl.testing import absltest
from fedjax.core import metrics
from fedjax.experimental import model
import haiku as hk
import jax
from jax.experimental import stax
import jax.numpy as jnp
import numpy as np


def _train_loss(batch, preds):
  num_classes = preds.shape[-1]
  log_preds = jax.nn.log_softmax(preds)
  one_hot_targets = jax.nn.one_hot(batch['y'], num_classes)
  return -jnp.sum(one_hot_targets * log_preds, axis=-1)


_eval_metric_fns = {
    'accuracy': lambda batch, preds: metrics.accuracy_fn(batch['y'], preds)
}


class ModelTest(absltest.TestCase):

  def check_model(self, model_):
    with self.subTest('init_params'):
      params = model_.init_params(jax.random.PRNGKey(0))
      num_params = sum(l.size for l in jax.tree_util.tree_leaves(params))
      self.assertEqual(num_params, 30)

    with self.subTest('apply_for_train'):
      batch = {
          'x': np.array([[1, 2], [3, 4], [5, 6]]),
          'y': np.array([7, 8, 9])
      }
      preds = model_.apply_for_train(params, batch, jax.random.PRNGKey(0))
      self.assertTupleEqual(preds.shape, (3, 10))

    with self.subTest('apply_for_eval'):
      preds = model_.apply_for_eval(params, batch)
      self.assertTupleEqual(preds.shape, (3, 10))

    with self.subTest('train_loss'):
      preds = model_.apply_for_train(params, batch, jax.random.PRNGKey(0))
      loss = model_.train_loss(batch, preds)
      self.assertTupleEqual(loss.shape, (3,))

  def test_create_model_from_haiku(self):

    def forward_pass(batch):
      return hk.Linear(10)(batch['x'])

    haiku_model = model.create_model_from_haiku(
        transformed_forward_pass=hk.transform(forward_pass),
        sample_batch={'x': np.ones((1, 2))},
        train_loss=_train_loss,
        eval_metric_fns=_eval_metric_fns)
    self.check_model(haiku_model)

  def test_create_model_from_stax(self):
    stax_init, stax_apply = stax.serial(stax.Dense(10))
    stax_model = model.create_model_from_stax(
        stax_init=stax_init,
        stax_apply=stax_apply,
        sample_shape=(-1, 2),
        train_loss=_train_loss,
        eval_metric_fns=_eval_metric_fns)
    self.check_model(stax_model)


if __name__ == '__main__':
  absltest.main()
