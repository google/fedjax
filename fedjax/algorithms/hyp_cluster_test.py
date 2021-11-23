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
"""Tests for hyp_cluster."""

from absl.testing import absltest
from fedjax.algorithms import hyp_cluster
from fedjax.core import client_datasets
from fedjax.core import metrics
from fedjax.core import models
from fedjax.core import optimizers
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt


class ClientTrainerTest(absltest.TestCase):

  OPTIMIZER = optimizers.sgd(1)

  def test_train_global_params(self):

    def grad(params, batch, rng):
      return 0.5 * params + jnp.mean(batch['x']) + jax.random.uniform(rng, [])

    rng_0 = jax.random.PRNGKey(0)
    rng_uniform_00 = jax.random.uniform(jax.random.split(rng_0)[1], [])
    rng_uniform_01 = jax.random.uniform(
        jax.random.split(jax.random.split(rng_0)[0])[1], [])
    rng_1 = jax.random.PRNGKey(1)
    rng_uniform_10 = jax.random.uniform(jax.random.split(rng_1)[1], [])

    params = jnp.array(1.)
    clients = [(b'0000', [{
        'x': jnp.array([0.125])
    }, {
        'x': jnp.array([0.25, 0.75])
    }], rng_0), (b'1001', [{
        'x': jnp.array([0, 1, 2])
    }], rng_1)]
    client_delta_params = dict(
        hyp_cluster.ClientDeltaTrainer(grad,
                                       self.OPTIMIZER).train_global_params(
                                           params, clients))
    self.assertCountEqual(client_delta_params, [b'0000', b'1001'])
    npt.assert_allclose(client_delta_params[b'0000'],
                        (0.5 * 1 + 0.125 + rng_uniform_00) +
                        (0.5 * (0.375 - rng_uniform_00) + 0.5 + rng_uniform_01))
    npt.assert_allclose(client_delta_params[b'1001'],
                        (0.5 * 1 + 1) + rng_uniform_10)

  def test_train_per_client_params(self):

    def grad(params, batch, rng):
      return 0.5 * params + jnp.mean(batch['x']) + jax.random.uniform(rng, [])

    rng_0 = jax.random.PRNGKey(0)
    rng_uniform_00 = jax.random.uniform(jax.random.split(rng_0)[1], [])
    rng_uniform_01 = jax.random.uniform(
        jax.random.split(jax.random.split(rng_0)[0])[1], [])
    rng_1 = jax.random.PRNGKey(1)
    rng_uniform_10 = jax.random.uniform(jax.random.split(rng_1)[1], [])

    clients = [
        (b'0000', [{
            'x': jnp.array([0.125])
        }, {
            'x': jnp.array([0.25, 0.75])
        }], rng_0, jnp.array(1.)),
        (b'1001', [{
            'x': jnp.array([0, 1, 2])
        }], rng_1, jnp.array(0.5)),
    ]
    client_delta_params = dict(
        hyp_cluster.ClientDeltaTrainer(
            grad, self.OPTIMIZER).train_per_client_params(clients))
    self.assertCountEqual(client_delta_params, [b'0000', b'1001'])
    npt.assert_allclose(client_delta_params[b'0000'],
                        (0.5 * 1 + 0.125 + rng_uniform_00) +
                        (0.5 * (0.375 - rng_uniform_00) + 0.5 + rng_uniform_01))
    npt.assert_allclose(client_delta_params[b'1001'],
                        (0.5 * 0.5 + 1) + rng_uniform_10)

  def test_return_params(self):

    def grad(params, batch, _):
      return 0.5 * params + jnp.mean(batch['x'])

    params = jnp.array(1.)
    clients = [(b'0000', [{
        'x': jnp.array([0.125])
    }, {
        'x': jnp.array([0.25, 0.75])
    }], jax.random.PRNGKey(0)),
               (b'1001', [{
                   'x': jnp.array([0, 1, 2])
               }], jax.random.PRNGKey(1))]
    client_delta_params = dict(
        hyp_cluster.ClientParamsTrainer(grad,
                                        self.OPTIMIZER).train_global_params(
                                            params, clients))
    npt.assert_equal(
        jax.device_get(client_delta_params), {
            b'0000': 1 - ((0.5 * 1 + 0.125) + (0.5 * 0.375 + 0.5)),
            b'1001': 1 - (0.5 * 1 + 1)
        })


class HypClusterTest(absltest.TestCase):

  def test_random_init(self):

    def init(rng):
      return jax.random.uniform(rng, [2])

    cluster_params = hyp_cluster.random_init(3, init, jax.random.PRNGKey(0))
    self.assertIsInstance(cluster_params, list)
    self.assertLen(cluster_params, 3)
    for i in cluster_params:
      self.assertIsInstance(i, jnp.ndarray)
      self.assertEqual(i.shape, (2,))
    self.assertTrue((cluster_params[0] != cluster_params[1]).any())
    self.assertTrue((cluster_params[0] != cluster_params[2]).any())
    self.assertTrue((cluster_params[1] != cluster_params[2]).any())

  def test_kmeans_init(self):
    functions_called = set()

    def init(rng):
      functions_called.add('init')
      return jax.random.uniform(rng)

    def apply_for_train(params, batch, rng):
      functions_called.add('apply_for_train')
      self.assertIsNotNone(rng)
      return params - batch['x']

    def train_loss(batch, out):
      functions_called.add('train_loss')
      return jnp.square(out) + batch['bias']

    def regularizer(params):
      del params
      functions_called.add('regularizer')
      return 0

    initializer = hyp_cluster.ModelKMeansInitializer(
        models.Model(
            init=init,
            apply_for_train=apply_for_train,
            apply_for_eval=None,
            train_loss=train_loss,
            eval_metrics={}), optimizers.sgd(0.5), regularizer)
    # Each client has 1 example, so it's very easy to reach minimal loss, at
    # which point the loss entirely depends on bias.
    clients = [
        (b'0',
         client_datasets.ClientDataset({
             'x': np.array([1.01]),
             'bias': np.array([-2.])
         }), jax.random.PRNGKey(1)),
        (b'1',
         client_datasets.ClientDataset({
             'x': np.array([3.02]),
             'bias': np.array([-1.])
         }), jax.random.PRNGKey(2)),
        (b'2',
         client_datasets.ClientDataset({
             'x': np.array([3.03]),
             'bias': np.array([1.])
         }), jax.random.PRNGKey(3)),
        (b'3',
         client_datasets.ClientDataset({
             'x': np.array([1.04]),
             'bias': np.array([2.])
         }), jax.random.PRNGKey(3)),
    ]
    train_batch_hparams = client_datasets.ShuffleRepeatBatchHParams(
        batch_size=1, num_epochs=5)
    eval_batch_hparams = client_datasets.PaddedBatchHParams(batch_size=2)
    # Using a rng that leads to b'0' being the initial center.
    cluster_params = initializer.cluster_params(
        num_clusters=3,
        rng=jax.random.PRNGKey(0),
        clients=clients,
        train_batch_hparams=train_batch_hparams,
        eval_batch_hparams=eval_batch_hparams)
    self.assertIsInstance(cluster_params, list)
    self.assertLen(cluster_params, 3)
    npt.assert_allclose(cluster_params, [1.01, 3.03, 1.04])
    self.assertCountEqual(
        functions_called,
        ['init', 'apply_for_train', 'train_loss', 'regularizer'])
    # Using a rng that leads to b'2' being the initial center.
    cluster_params = initializer.cluster_params(
        num_clusters=3,
        rng=jax.random.PRNGKey(1),
        clients=clients,
        train_batch_hparams=train_batch_hparams,
        eval_batch_hparams=eval_batch_hparams)
    self.assertIsInstance(cluster_params, list)
    self.assertLen(cluster_params, 3)
    npt.assert_allclose(cluster_params, [3.03, 1.04, 1.04])

  def test_maximization_step(self):
    # L1 distance from centers.
    def per_example_loss(params, batch, rng):
      # Randomly flip the center to test rng behavior.
      sign = jax.random.bernoulli(rng) * 2 - 1
      return jnp.abs(params * sign - batch['x'])

    def regularizer(params):
      return 0.01 * jnp.sum(jnp.abs(params))

    evaluator = models.AverageLossEvaluator(per_example_loss, regularizer)
    cluster_params = [jnp.array(0.), jnp.array(-1.), jnp.array(2.)]
    # Batch size is chosen so that we run 1 or 2 batches.
    batch_hparams = client_datasets.PaddedBatchHParams(batch_size=2)
    # Special seeds:
    # - No flip in first 2 steps for all 3 clusters: 0;
    # - Flip all in first 2 steps for all 3 clusters: 16;
    # - No flip then flip all for all 3 clusters: 68;
    # - Flips only cluster 1 in first 2 steps: 106.
    clients = [
        # No flip in first 2 steps for all 3 clusters.
        (b'near0', client_datasets.ClientDataset({'x': np.array([0.1])}),
         jax.random.PRNGKey(0)),
        # Flip all in first 2 steps for all 3 clusters.
        (b'near-1',
         client_datasets.ClientDataset({'x': np.array([0.9, 1.1, 1.3])}),
         jax.random.PRNGKey(16)),
        # No flip then flip all for all 3 clusters.
        (b'near2',
         client_datasets.ClientDataset({'x': np.array([1.9, 2.1, -2.1])}),
         jax.random.PRNGKey(68)),
        # Flips only cluster 1 in first 2 steps.
        (b'near1',
         client_datasets.ClientDataset({'x': np.array([0.9, 1.1, 1.3])}),
         jax.random.PRNGKey(106)),
    ]

    cluster_losses = hyp_cluster._cluster_losses(
        evaluator=evaluator,
        cluster_params=cluster_params,
        clients=clients,
        batch_hparams=batch_hparams)
    self.assertCountEqual(cluster_losses,
                          [b'near0', b'near-1', b'near2', b'near1'])
    npt.assert_allclose(cluster_losses[b'near0'],
                        np.array([0.1, 1.1 + 0.01, 1.9 + 0.02]))
    npt.assert_allclose(cluster_losses[b'near-1'],
                        np.array([1.1, 0.5 / 3 + 0.01, 3.1 + 0.02]))
    npt.assert_allclose(
        cluster_losses[b'near2'],
        np.array([6.1 / 3, 9.1 / 3 + 0.01, 0.1 + 0.02]),
        rtol=1e-6)
    npt.assert_allclose(cluster_losses[b'near1'],
                        np.array([1.1, 0.5 / 3 + 0.01, 0.9 + 0.02]))

    client_cluster_ids = hyp_cluster.maximization_step(
        evaluator=evaluator,
        cluster_params=cluster_params,
        clients=clients,
        batch_hparams=batch_hparams)
    self.assertDictEqual(client_cluster_ids, {
        b'near0': 0,
        b'near-1': 1,
        b'near2': 2,
        b'near1': 1
    })

  def test_hyp_cluster_evaluator(self):
    functions_called = set()

    def apply_for_eval(params, batch):
      functions_called.add('apply_for_eval')
      score = params * batch['x']
      return jnp.stack([-score, score], axis=-1)

    def apply_for_train(params, batch, rng):
      functions_called.add('apply_for_train')
      self.assertIsNotNone(rng)
      return params * batch['x']

    def train_loss(batch, out):
      functions_called.add('train_loss')
      return jnp.abs(batch['y'] * 2 - 1 - out)

    def regularizer(params):
      # Just to check regularizer is called.
      del params
      functions_called.add('regularizer')
      return 0

    evaluator = hyp_cluster.HypClusterEvaluator(
        models.Model(
            init=None,
            apply_for_eval=apply_for_eval,
            apply_for_train=apply_for_train,
            train_loss=train_loss,
            eval_metrics={'accuracy': metrics.Accuracy()}), regularizer)

    cluster_params = [jnp.array(1.), jnp.array(-1.)]
    train_clients = [
        # Evaluated using cluster 0.
        (b'0',
         client_datasets.ClientDataset({
             'x': np.array([3., 2., 1.]),
             'y': np.array([1, 1, 0])
         }), jax.random.PRNGKey(0)),
        # Evaluated using cluster 1.
        (b'1',
         client_datasets.ClientDataset({
             'x': np.array([0.9, -0.9, 0.8, -0.8, -0.3]),
             'y': np.array([0, 1, 0, 1, 0])
         }), jax.random.PRNGKey(1)),
    ]
    # Test clients are generated from train_clients by swapping client ids and
    # then flipping labels.
    test_clients = [
        # Evaluated using cluster 0.
        (b'0',
         client_datasets.ClientDataset({
             'x': np.array([0.9, -0.9, 0.8, -0.8, -0.3]),
             'y': np.array([1, 0, 1, 0, 1])
         })),
        # Evaluated using cluster 1.
        (b'1',
         client_datasets.ClientDataset({
             'x': np.array([3., 2., 1.]),
             'y': np.array([0, 0, 1])
         })),
    ]
    for batch_size in [1, 2, 4]:
      with self.subTest(f'batch_size = {batch_size}'):
        batch_hparams = client_datasets.PaddedBatchHParams(
            batch_size=batch_size)
        metric_values = dict(
            evaluator.evaluate_clients(
                cluster_params=cluster_params,
                train_clients=train_clients,
                test_clients=test_clients,
                batch_hparams=batch_hparams))
        self.assertCountEqual(metric_values, [b'0', b'1'])
        self.assertCountEqual(metric_values[b'0'], ['accuracy'])
        npt.assert_allclose(metric_values[b'0']['accuracy'], 4 / 5)
        self.assertCountEqual(metric_values[b'1'], ['accuracy'])
        npt.assert_allclose(metric_values[b'1']['accuracy'], 2 / 3)
    self.assertCountEqual(
        functions_called,
        ['apply_for_train', 'train_loss', 'apply_for_eval', 'regularizer'])

  def test_expectation_step(self):

    def per_example_loss(params, batch, rng):
      self.assertIsNotNone(rng)
      return jnp.square(params - batch['x'])

    trainer = hyp_cluster.ClientDeltaTrainer(
        models.grad(per_example_loss), optimizers.sgd(0.5))
    batch_hparams = client_datasets.ShuffleRepeatBatchHParams(
        batch_size=1, num_epochs=5)

    cluster_params = [jnp.array(1.), jnp.array(-1.), jnp.array(3.14)]
    client_cluster_ids = {b'0': 0, b'1': 0, b'2': 1, b'3': 1, b'4': 0}
    # RNGs are not actually used.
    clients = [
        (b'0', client_datasets.ClientDataset({'x': np.array([1.1])}),
         jax.random.PRNGKey(0)),
        (b'1', client_datasets.ClientDataset({'x': np.array([0.9, 0.9])}),
         jax.random.PRNGKey(1)),
        (b'2', client_datasets.ClientDataset({'x': np.array([-1.1])}),
         jax.random.PRNGKey(2)),
        (b'3',
         client_datasets.ClientDataset({'x': np.array([-0.9, -0.9, -0.9])}),
         jax.random.PRNGKey(3)),
        (b'4', client_datasets.ClientDataset({'x': np.array([-0.1])}),
         jax.random.PRNGKey(4)),
    ]
    cluster_delta_params = hyp_cluster.expectation_step(
        trainer=trainer,
        cluster_params=cluster_params,
        client_cluster_ids=client_cluster_ids,
        clients=clients,
        batch_hparams=batch_hparams)
    self.assertIsInstance(cluster_delta_params, list)
    self.assertLen(cluster_delta_params, 3)
    npt.assert_allclose(cluster_delta_params[0], (-0.1 + 0.1 * 2 + 1.1) / 4)
    npt.assert_allclose(cluster_delta_params[1], (0.1 - 0.1 * 3) / 4, rtol=1e-6)
    self.assertIsNone(cluster_delta_params[2])

  def test_hyp_cluster(self):
    functions_called = set()

    def per_example_loss(params, batch, rng):
      self.assertIsNotNone(rng)
      functions_called.add('per_example_loss')
      return jnp.square(params - batch['x'])

    def regularizer(params):
      del params
      functions_called.add('regularizer')
      return 0

    client_optimizer = optimizers.sgd(0.5)
    server_optimizer = optimizers.sgd(0.25)

    maximization_batch_hparams = client_datasets.PaddedBatchHParams(
        batch_size=2)
    expectation_batch_hparams = client_datasets.ShuffleRepeatBatchHParams(
        batch_size=1, num_epochs=5)

    algorithm = hyp_cluster.hyp_cluster(
        per_example_loss=per_example_loss,
        client_optimizer=client_optimizer,
        server_optimizer=server_optimizer,
        maximization_batch_hparams=maximization_batch_hparams,
        expectation_batch_hparams=expectation_batch_hparams,
        regularizer=regularizer)

    init_state = algorithm.init([jnp.array(1.), jnp.array(-1.)])
    # Nothing happens with empty data.
    no_op_state, diagnostics = algorithm.apply(init_state, clients=[])
    npt.assert_array_equal(init_state.cluster_params,
                           no_op_state.cluster_params)
    self.assertEmpty(diagnostics)
    # Some actual training. PRNGKeys are not actually used.
    clients = [
        (b'0', client_datasets.ClientDataset({'x': np.array([1.1])}),
         jax.random.PRNGKey(0)),
        (b'1', client_datasets.ClientDataset({'x': np.array([0.9, 0.9])}),
         jax.random.PRNGKey(1)),
        (b'2', client_datasets.ClientDataset({'x': np.array([-1.1])}),
         jax.random.PRNGKey(2)),
        (b'3',
         client_datasets.ClientDataset({'x': np.array([-0.9, -0.9, -0.9])}),
         jax.random.PRNGKey(3)),
    ]
    next_state, diagnostics = algorithm.apply(init_state, clients)
    npt.assert_equal(
        diagnostics, {
            b'0': {
                'cluster_id': 0
            },
            b'1': {
                'cluster_id': 0
            },
            b'2': {
                'cluster_id': 1
            },
            b'3': {
                'cluster_id': 1
            },
        })
    cluster_params = next_state.cluster_params
    self.assertIsInstance(cluster_params, list)
    self.assertLen(cluster_params, 2)
    npt.assert_allclose(cluster_params[0], [1. - 0.25 * 0.1 / 3])
    npt.assert_allclose(cluster_params[1], [-1. + 0.25 * 0.2 / 4])
    self.assertCountEqual(functions_called, ['per_example_loss', 'regularizer'])


if __name__ == '__main__':
  absltest.main()
