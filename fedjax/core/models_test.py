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
"""Tests for fedjax.core.models."""

from absl.testing import absltest

from fedjax.core import metrics
from fedjax.core import models

import haiku as hk
import jax
try:
  from jax.example_libraries import stax
except ModuleNotFoundError:
  from jax.experimental import stax  # pytype: disable=import-error
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

train_loss = lambda b, p: metrics.unreduced_cross_entropy_loss(b['y'], p)
eval_metrics = {'accuracy': metrics.Accuracy()}


class ModelTest(absltest.TestCase):

  def check_model(self, model):
    with self.subTest('init'):
      params = model.init(jax.random.PRNGKey(0))
      num_params = sum(l.size for l in jax.tree_util.tree_leaves(params))
      self.assertEqual(num_params, 30)

    with self.subTest('apply_for_train'):
      batch = {
          'x': np.array([[1, 2], [3, 4], [5, 6]]),
          'y': np.array([7, 8, 9])
      }
      preds = model.apply_for_train(params, batch, jax.random.PRNGKey(0))
      self.assertTupleEqual(preds.shape, (3, 10))

    with self.subTest('apply_for_eval'):
      preds = model.apply_for_eval(params, batch)
      self.assertTupleEqual(preds.shape, (3, 10))

    with self.subTest('train_loss'):
      preds = model.apply_for_train(params, batch, jax.random.PRNGKey(0))
      loss = model.train_loss(batch, preds)
      self.assertTupleEqual(loss.shape, (3,))

  def test_create_model_from_haiku(self):

    def forward_pass(batch):
      return hk.Linear(10)(batch['x'])

    haiku_model = models.create_model_from_haiku(
        transformed_forward_pass=hk.transform(forward_pass),
        sample_batch={'x': jnp.ones((1, 2))},
        train_loss=train_loss,
        eval_metrics=eval_metrics)
    self.check_model(haiku_model)

  def test_create_model_from_stax(self):
    stax_init, stax_apply = stax.serial(stax.Dense(10))
    stax_model = models.create_model_from_stax(
        stax_init=stax_init,
        stax_apply=stax_apply,
        sample_shape=(-1, 2),
        train_loss=train_loss,
        eval_metrics=eval_metrics)
    self.check_model(stax_model)

  def test_evaluate_model(self):
    # Mock out Model.
    model = models.Model(
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
    eval_results = models.evaluate_model(model, params, batches)
    self.assertEqual(eval_results['accuracy'], 0.625)  # 5 / 8.
    self.assertAlmostEqual(eval_results['loss'], 0.8419596, places=6)

  def test_evaluate_global_params(self):
    # Mock out Model.
    model = models.Model(
        init=lambda rng: None,  # Unused.
        apply_for_train=lambda params, batch, rng: None,  # Unused.
        apply_for_eval=lambda params, batch: batch.get('pred') + params,
        train_loss=lambda batch, preds: None,  # Unused.
        eval_metrics={
            'accuracy': metrics.Accuracy(),
            'loss': metrics.CrossEntropyLoss(),
        })
    params = jnp.array([1, -1])
    clients = [
        (b'0000', [{
            'y': np.array([1, 0, 1]),
            'pred': np.array([[0.2, 1.4], [1.3, 1.1], [-0.7, 4.2]])
        }, {
            'y': np.array([0, 1, 1]),
            'pred': np.array([[0.2, 1.4], [1.3, 1.1], [-0.7, 4.2]])
        }]),
        (b'1001', [{
            'y': np.array([0, 1]),
            'pred': np.array([[0.2, 1.4], [1.3, 1.1]])
        }]),
    ]
    eval_results = dict(
        models.ModelEvaluator(model).evaluate_global_params(params, clients))
    self.assertCountEqual(eval_results, [b'0000', b'1001'])
    self.assertCountEqual(eval_results[b'0000'], ['accuracy', 'loss'])
    self.assertAlmostEqual(eval_results[b'0000']['accuracy'], 4 / 6)
    self.assertAlmostEqual(eval_results[b'0000']['loss'], 0.67658216)
    self.assertCountEqual(eval_results[b'1001'], ['accuracy', 'loss'])
    self.assertEqual(eval_results[b'1001']['accuracy'], 1 / 2)
    self.assertAlmostEqual(eval_results[b'1001']['loss'], 1.338092, places=6)

  def test_evaluate_per_client_params(self):
    # Mock out Model.
    model = models.Model(
        init=lambda rng: None,  # Unused.
        apply_for_train=lambda params, batch, rng: None,  # Unused.
        apply_for_eval=lambda params, batch: batch.get('pred') + params,
        train_loss=lambda batch, preds: None,  # Unused.
        eval_metrics={
            'accuracy': metrics.Accuracy(),
            'loss': metrics.CrossEntropyLoss(),
        })
    clients = [
        (b'0000', [{
            'y': np.array([1, 0, 1]),
            'pred': np.array([[0.2, 1.4], [1.3, 1.1], [-0.7, 4.2]])
        }, {
            'y': np.array([0, 1, 1]),
            'pred': np.array([[0.2, 1.4], [1.3, 1.1], [-0.7, 4.2]])
        }], jnp.array([1, -1])),
        (b'1001', [{
            'y': np.array([0, 1]),
            'pred': np.array([[1.2, 0.4], [2.3, 0.1]])
        }], jnp.array([0, 0])),
    ]
    eval_results = dict(
        models.ModelEvaluator(model).evaluate_per_client_params(clients))
    self.assertCountEqual(eval_results, [b'0000', b'1001'])
    self.assertCountEqual(eval_results[b'0000'], ['accuracy', 'loss'])
    npt.assert_allclose(eval_results[b'0000']['accuracy'], 4 / 6)
    npt.assert_allclose(eval_results[b'0000']['loss'], 0.67658216)
    self.assertCountEqual(eval_results[b'1001'], ['accuracy', 'loss'])
    npt.assert_allclose(eval_results[b'1001']['accuracy'], 1 / 2)
    npt.assert_allclose(eval_results[b'1001']['loss'], 1.338092)

  def test_model_per_example_loss(self):
    # Mock out Model.
    model = models.Model(
        init=lambda rng: None,  # Unused.
        apply_for_train=lambda params, batch, rng: batch['x'] * params + rng,
        apply_for_eval=lambda params, batch: None,  # Unused.
        train_loss=lambda batch, preds: jnp.abs(batch['y'] - preds),
        eval_metrics={}  # Unused
    )

    params = jnp.array(2.)
    batch = {'x': jnp.array([1., -1., 1.]), 'y': jnp.array([0.1, -0.1, -0.1])}
    rng = jnp.array(0.5)

    loss = models.model_per_example_loss(model)(params, batch, rng)
    npt.assert_allclose(loss, [2.4, 1.4, 2.6])

  def test_model_grad(self):
    # Mock out Model.
    model = models.Model(
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
      grads = models.model_grad(model)(params, batch, rng)
      npt.assert_allclose(grads, (2.4 + 1.4 + 2.6) / 3)

    with self.subTest('has regularizer'):
      grads = models.model_grad(model, jnp.abs)(params, batch, rng)
      npt.assert_allclose(grads, (2.4 + 1.4 + 2.6) / 3 + 1)

    with self.subTest('has mask'):
      grads = models.model_grad(model)(params, {
          **batch, '__mask__': jnp.array([True, False, True])
      }, rng)
      npt.assert_allclose(grads, (2.4 + 2.6) / 2)

    with self.subTest('has regularizer, has mask'):
      grads = models.model_grad(model, jnp.abs)(params, {
          **batch, '__mask__': jnp.array([True, False, True])
      }, rng)
      npt.assert_allclose(grads, (2.4 + 2.6) / 2 + 1)


class AverageLossTest(absltest.TestCase):

  def test_evaluate_average_loss(self):

    def per_example_loss(params, batch, rng):
      return params + batch['x'] + jax.random.uniform(rng, [])

    def regularizer(params):
      return 0.5 * jnp.sum(jnp.square(params))

    params = jnp.array(1)

    rng = jax.random.PRNGKey(0)
    rng_uniform_0 = jax.random.uniform(jax.random.split(rng)[1], [])
    rng_uniform_1 = jax.random.uniform(
        jax.random.split(jax.random.split(rng)[0])[1], [])
    rng_term = (rng_uniform_0 * 2 + rng_uniform_1 * 3) / 5

    with self.subTest('no mask, no regularizer'):
      batches = [{'x': jnp.array([1, 2])}, {'x': jnp.array([3, 4, 5])}]
      average_loss = models.evaluate_average_loss(
          params=params,
          batches=batches,
          rng=rng,
          per_example_loss=per_example_loss)
      npt.assert_allclose(average_loss, 4 + rng_term, rtol=1e-6)

    with self.subTest('no mask, has regularizer'):
      batches = [{'x': jnp.array([1, 2])}, {'x': jnp.array([3, 4, 5])}]
      average_loss = models.evaluate_average_loss(
          params=params,
          batches=batches,
          rng=rng,
          per_example_loss=per_example_loss,
          regularizer=regularizer)
      npt.assert_allclose(average_loss, 4.5 + rng_term, rtol=1e-6)

    with self.subTest('has mask, no regularizer'):
      batches = [{
          'x': jnp.array([1, 2, 10]),
          '__mask__': jnp.array([True, True, False])
      }, {
          'x': jnp.array([3, 4, 5])
      }]
      average_loss = models.evaluate_average_loss(
          params=params,
          batches=batches,
          rng=rng,
          per_example_loss=per_example_loss)
      npt.assert_allclose(average_loss, 4 + rng_term, rtol=1e-6)

  def test_evaluate_global_params(self):

    def per_example_loss(params, batch, rng):
      return params + batch['x'] + jax.random.uniform(rng, [])

    def regularizer(params):
      return 0.5 * jnp.sum(jnp.square(params))

    params = jnp.array(1)

    rng_0 = jax.random.PRNGKey(0)
    rng_uniform_00 = jax.random.uniform(jax.random.split(rng_0)[1], [])
    rng_term_0 = rng_uniform_00
    rng_1 = jax.random.PRNGKey(1)
    rng_uniform_10 = jax.random.uniform(jax.random.split(rng_1)[1], [])
    rng_uniform_11 = jax.random.uniform(
        jax.random.split(jax.random.split(rng_1)[0])[1], [])
    rng_term_1 = (rng_uniform_10 * 2 + rng_uniform_11) / 3

    with self.subTest('no mask, no regularizer'):
      clients = [
          (b'0000', [{
              'x': jnp.array([1, 2])
          }], rng_0),
          (b'1001', [{
              'x': jnp.array([3, 4])
          }, {
              'x': jnp.array([5])
          }], rng_1),
      ]
      average_loss = dict(
          models.AverageLossEvaluator(per_example_loss).evaluate_global_params(
              params=params, clients=clients))
      npt.assert_equal(average_loss, {
          b'0000': np.array(2.5) + rng_term_0,
          b'1001': np.array(5) + rng_term_1
      })

    with self.subTest('no mask, has regularizer'):
      clients = [
          (b'0000', [{
              'x': jnp.array([1, 2])
          }], rng_0),
          (b'1001', [{
              'x': jnp.array([3, 4])
          }, {
              'x': jnp.array([5])
          }], rng_1),
      ]
      average_loss = dict(
          models.AverageLossEvaluator(per_example_loss,
                                      regularizer).evaluate_global_params(
                                          params=params, clients=clients))
      npt.assert_equal(average_loss, {
          b'0000': np.array(3) + rng_term_0,
          b'1001': np.array(5.5) + rng_term_1
      })

    with self.subTest('has mask, no regularizer'):
      clients = [
          (b'0000', [{
              'x': jnp.array([1, 2])
          }], rng_0),
          (b'1001', [{
              'x': jnp.array([3, 4])
          }, {
              'x': jnp.array([5, 10]),
              '__mask__': jnp.array([True, False])
          }], rng_1),
      ]
      average_loss = dict(
          models.AverageLossEvaluator(per_example_loss).evaluate_global_params(
              params=params, clients=clients))
      npt.assert_equal(average_loss, {
          b'0000': np.array(2.5) + rng_term_0,
          b'1001': np.array(5) + rng_term_1
      })

  def test_evaluate_per_client_params(self):

    def per_example_loss(params, batch, rng):
      return params + batch['x'] + jax.random.uniform(rng, [])

    def regularizer(params):
      return 0.5 * jnp.sum(jnp.square(params))

    rng_0 = jax.random.PRNGKey(0)
    rng_uniform_00 = jax.random.uniform(jax.random.split(rng_0)[1], [])
    rng_term_0 = rng_uniform_00
    rng_1 = jax.random.PRNGKey(1)
    rng_uniform_10 = jax.random.uniform(jax.random.split(rng_1)[1], [])
    rng_uniform_11 = jax.random.uniform(
        jax.random.split(jax.random.split(rng_1)[0])[1], [])
    rng_term_1 = (rng_uniform_10 * 2 + rng_uniform_11) / 3

    with self.subTest('no mask, no regularizer'):
      clients = [
          (b'0000', [{
              'x': jnp.array([2, 3])
          }], rng_0, jnp.array(0)),
          (b'1001', [{
              'x': jnp.array([3, 4])
          }, {
              'x': jnp.array([5])
          }], rng_1, jnp.array(1)),
      ]
      average_loss = dict(
          models.AverageLossEvaluator(
              per_example_loss).evaluate_per_client_params(clients=clients))
      npt.assert_equal(average_loss, {
          b'0000': np.array(2.5) + rng_term_0,
          b'1001': np.array(5) + rng_term_1
      })

    with self.subTest('no mask, has regularizer'):
      clients = [
          (b'0000', [{
              'x': jnp.array([2, 3])
          }], rng_0, jnp.array(0)),
          (b'1001', [{
              'x': jnp.array([3, 4])
          }, {
              'x': jnp.array([5])
          }], rng_1, jnp.array(1)),
      ]
      average_loss = dict(
          models.AverageLossEvaluator(
              per_example_loss,
              regularizer).evaluate_per_client_params(clients=clients))
      npt.assert_equal(average_loss, {
          b'0000': np.array(2.5) + rng_term_0,
          b'1001': np.array(5.5) + rng_term_1
      })

    with self.subTest('has mask, no regularizer'):
      clients = [
          (b'0000', [{
              'x': jnp.array([2, 3])
          }], rng_0, jnp.array(0)),
          (b'1001', [{
              'x': jnp.array([3, 4])
          }, {
              'x': jnp.array([5, 10]),
              '__mask__': jnp.array([True, False])
          }], rng_1, jnp.array(1)),
      ]
      average_loss = dict(
          models.AverageLossEvaluator(
              per_example_loss).evaluate_per_client_params(clients=clients))
      npt.assert_equal(average_loss, {
          b'0000': np.array(2.5) + rng_term_0,
          b'1001': np.array(5) + rng_term_1
      })


if __name__ == '__main__':
  absltest.main()
