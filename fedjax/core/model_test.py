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
"""Tests for fedjax.core.model."""

import collections

from fedjax.core import metrics
from fedjax.core import model
import haiku as hk
import jax
from jax.experimental import stax
import numpy as np
import tensorflow as tf


def _loss(batch, preds):
  return metrics.cross_entropy_loss_fn(targets=batch['y'], preds=preds)


def _accuracy(batch, preds):
  return metrics.accuracy_fn(targets=batch['y'], preds=preds)


def _create_haiku_model(num_classes,
                        sample_batch,
                        non_trainable_module_names=()):
  """Creates toy haiku model."""

  def forward_pass(batch):
    network = hk.Sequential(
        [hk.Linear(2 * num_classes),
         hk.Linear(num_classes)])
    return network(batch['x'])

  transformed_forward_pass = hk.transform(forward_pass)
  metrics_fn_map = collections.OrderedDict(accuracy=_accuracy)
  return model.create_model_from_haiku(
      transformed_forward_pass=transformed_forward_pass,
      sample_batch=sample_batch,
      loss_fn=_loss,
      metrics_fn_map=metrics_fn_map,
      non_trainable_module_names=non_trainable_module_names)


def _create_stax_model(num_classes, sample_shape):
  """Creates toy stax model."""
  stax_init_fn, stax_apply_fn = stax.serial(stax.Flatten,
                                            stax.Dense(2 * num_classes),
                                            stax.Dense(num_classes))
  metrics_fn_map = collections.OrderedDict(accuracy=_accuracy)
  return model.create_model_from_stax(
      stax_init_fn=stax_init_fn,
      stax_apply_fn=stax_apply_fn,
      sample_shape=sample_shape,
      loss_fn=_loss,
      metrics_fn_map=metrics_fn_map)


class StaxModelTest(tf.test.TestCase):

  def create_model(self):
    return _create_stax_model(self._num_classes, (-1, self._feature_dim))

  def setUp(self):
    super().setUp()
    self._batch_size = 5
    self._feature_dim = 10
    self._num_classes = 2
    self._rng_seq = hk.PRNGSequence(0)
    self._batch = collections.OrderedDict(
        x=np.ones((self._batch_size, self._feature_dim)),
        y=np.ones((self._batch_size,)))
    self._model = self.create_model()

  def test_init_params(self):
    params = self._model.init_params(rng=next(self._rng_seq))
    self.assertEqual(hk.data_structures.tree_size(params), 54)

  def test_backward_pass(self):
    params = self._model.init_params(rng=next(self._rng_seq))
    output = self._model.backward_pass(
        params=params, batch=self._batch, rng=next(self._rng_seq))
    tf.nest.map_structure(lambda a, b: self.assertTupleEqual(a.shape, b.shape),
                          params, output.grads)
    self.assertGreaterEqual(output.loss, 0.)
    self.assertEqual(output.num_examples, self._batch_size)

  def test_evaluate(self):
    params = self._model.init_params(rng=next(self._rng_seq))
    eval_metrics = self._model.evaluate(params=params, batch=self._batch)

    self.assertContainsSubset(
        ['num_examples', 'regularizer', 'loss', 'accuracy'],
        list(eval_metrics.keys()))

  def test_train(self):
    params = self._model.init_params(rng=next(self._rng_seq))
    prev_loss = self._model.evaluate(
        params=params, batch=self._batch)['loss'].result()
    for _ in range(5):
      output = self._model.backward_pass(
          params=params, batch=self._batch, rng=next(self._rng_seq))
      params = jax.tree_multimap(lambda p, g: p - 0.01 * g, params,
                                 output.grads)
      loss = self._model.evaluate(
          params=params, batch=self._batch)['loss'].result()
      self.assertLess(loss, prev_loss)
      prev_loss = loss


class HaikuModelTest(StaxModelTest):

  def create_model(self):
    return _create_haiku_model(self._num_classes, self._batch)

  def test_train_with_some_frozen(self):
    non_trainable_module_names = ['linear_1']
    model_ = _create_haiku_model(self._num_classes, self._batch,
                                 non_trainable_module_names)
    params = model_.init_params(rng=next(self._rng_seq))
    predicate = lambda module_name, *_: module_name in non_trainable_module_names
    init_non_trainable_params, init_trainable_params = (
        hk.data_structures.partition(predicate, params))

    for _ in range(5):
      output = model_.backward_pass(
          params=params, batch=self._batch, rng=next(self._rng_seq))
      params = jax.tree_multimap(lambda p, g: p - 0.01 * g, params,
                                 output.grads)

    non_trainable_params, trainable_params = (
        hk.data_structures.partition(predicate, params))
    jax.tree_multimap(self.assertAllEqual, non_trainable_params,
                      init_non_trainable_params)
    jax.tree_multimap(self.assertNotAllEqual, trainable_params,
                      init_trainable_params)


if __name__ == '__main__':
  tf.test.main()
