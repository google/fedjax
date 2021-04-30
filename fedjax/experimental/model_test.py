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

from fedjax.experimental import metrics
from fedjax.experimental import model

import haiku as hk
import jax
from jax.experimental import stax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

train_loss = lambda b, p: metrics.unreduced_cross_entropy_loss(b['y'], p)
eval_metrics = {'accuracy': metrics.Accuracy()}


class ModelTest(absltest.TestCase):

  def check_model(self, model_):
    with self.subTest('init'):
      params = model_.init(jax.random.PRNGKey(0))
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
        train_loss=train_loss,
        eval_metrics=eval_metrics)
    self.check_model(haiku_model)

  def test_create_model_from_stax(self):
    stax_init, stax_apply = stax.serial(stax.Dense(10))
    stax_model = model.create_model_from_stax(
        stax_init=stax_init,
        stax_apply=stax_apply,
        sample_shape=(-1, 2),
        train_loss=train_loss,
        eval_metrics=eval_metrics)
    self.check_model(stax_model)

  def test_evaluate_model(self):
    # Mock out Model.
    model_ = model.Model.new(
        init=lambda rng: None,  # Unused.
        apply_for_train=lambda params, batch, rng: None,  # Unused.
        apply_for_eval=lambda params, batch: batch.get('pred'),
        train_loss=lambda batch, preds: None,  # Unused.
        eval_metrics={
            'accuracy': metrics.Accuracy(),
            'loss': metrics.CrossEntropyLoss(),
        })
    params = jnp.array(3.14)  # Unused.
    batches = [
        {
            'y': np.array([1, 0, 1]),
            'pred': np.array([[1.2, 0.4], [2.3, 0.1], [0.3, 3.2]])
        },
        {
            'y': np.array([0, 1, 1]),
            'pred': np.array([[1.2, 0.4], [2.3, 0.1], [0.3, 3.2]])
        },
        {
            'y': np.array([0, 1]),
            'pred': np.array([[1.2, 0.4], [2.3, 0.1]])
        },
    ]
    eval_results = model.evaluate_model(model_, params, batches)
    self.assertEqual(eval_results['accuracy'], 0.625)  # 5 / 8.
    self.assertAlmostEqual(eval_results['loss'], 0.8419596)

  def test_model_grad(self):
    # Mock out Model.
    model_ = model.Model.new(
        init=lambda rng: None,  # Unused.
        apply_for_train=lambda params, batch, rng: batch['x'] * params + rng,
        apply_for_eval=lambda params, batch: None,  # Unused.
        train_loss=lambda batch, preds: jnp.square(batch['y'] - preds) / 2,
        eval_metrics={}  # Unused
    )

    params = jnp.array(2.)
    batch = {'x': jnp.array([1., -1., 1.]), 'y': jnp.array([0.1, -0.1, -0.1])}
    rng = jnp.array(0.5)

    with self.subTest('no regularizer'):
      grads = model.model_grad(model_)(params, batch, rng)
      npt.assert_allclose(grads, (2.4 + 1.4 + 2.6) / 3)

    with self.subTest('has regularizer'):
      grads = model.model_grad(model_, jnp.abs)(params, batch, rng)
      npt.assert_allclose(grads, (2.4 + 1.4 + 2.6) / 3 + 1)

    with self.subTest('has mask'):
      grads = model.model_grad(model_)(params, {
          **batch, '__mask__': jnp.array([True, False, True])
      }, rng)
      npt.assert_allclose(grads, (2.4 + 2.6) / 2)

    with self.subTest('has regularizer, has mask'):
      grads = model.model_grad(model_, jnp.abs)(params, {
          **batch, '__mask__': jnp.array([True, False, True])
      }, rng)
      npt.assert_allclose(grads, (2.4 + 2.6) / 2 + 1)


if __name__ == '__main__':
  absltest.main()
