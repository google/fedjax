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
"""Tests for fedjax.core.client_trainer."""

import os

from absl.testing import flagsaver
from absl.testing import parameterized

from fedjax.core import client_trainer
from fedjax.core import dataset_util
from fedjax.core import evaluation_util
from fedjax.core import optimizer
from fedjax.core import test_util

import jax
from jax.lib import xla_bridge
import jax.numpy as jnp
import tensorflow as tf


def setUpModule():
  """Run all tests with 8 CPU devices."""
  global prev_xla_flags  # pylint: disable=global-variable-undefined
  prev_xla_flags = os.getenv('XLA_FLAGS')
  flags_str = prev_xla_flags or ''
  # Don't override user-specified device count, or other XLA flags.
  if 'xla_force_host_platform_device_count' not in flags_str:
    os.environ['XLA_FLAGS'] = (
        flags_str + ' --xla_force_host_platform_device_count=8')
  # Clear any cached backends so new CPU backend will pick up the env var.
  xla_bridge.get_backend.cache_clear()


def tearDownModule():
  """Reset to previous configuration in case other test modules will be run."""
  if prev_xla_flags is None:
    del os.environ['XLA_FLAGS']
  else:
    os.environ['XLA_FLAGS'] = prev_xla_flags
  xla_bridge.get_backend.cache_clear()


class DefaultClientTrainerTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._federated_algorithm = test_util.MockFederatedAlgorithm(
        num_clients=5, num_examples=20)
    self._federated_data = self._federated_algorithm.federated_data
    self._client_data_hparams = dataset_util.ClientDataHParams(batch_size=3)
    self._client_dataset = self._federated_data.create_tf_dataset_for_client(
        self._federated_data.client_ids[0]).batch(10)
    self._trainer = client_trainer.DefaultClientTrainer(
        self._federated_algorithm.model,
        optimizer.get_optimizer(
            optimizer.OptimizerName.SGD, learning_rate=0.01))

  def init_state(self):
    algo_state = self._federated_algorithm.init_state()
    return self._trainer.init_state(algo_state.params)

  def test_one_step(self):
    state = self.init_state()
    batch = next(self._client_dataset.as_numpy_iterator())
    rng = next(self._federated_algorithm._rng_seq)

    prev_metrics = self._federated_algorithm.model.evaluate(state.params, batch)
    state = self._trainer.one_step(state, batch, rng)
    metrics = self._federated_algorithm.model.evaluate(state.params, batch)

    self.assertEqual(state.num_examples, 10)
    self.assertLess(metrics['loss'].result(), prev_metrics['loss'].result())

  def test_loop(self):
    state = self.init_state()
    examples = zip(self._client_dataset.as_numpy_iterator(),
                   self._federated_algorithm._rng_seq)

    prev_metrics = evaluation_util.evaluate_single_client(
        self._client_dataset, self._federated_algorithm.model, state.params)
    state = self._trainer.loop(state, examples)
    metrics = evaluation_util.evaluate_single_client(
        self._client_dataset, self._federated_algorithm.model, state.params)

    self.assertEqual(state.num_examples, 20)
    self.assertLess(metrics['loss'], prev_metrics['loss'])

  def test_train_single_client_tf_dataset(self):
    state = client_trainer.train_single_client(
        dataset=self._client_dataset,
        client_trainer=self._trainer,
        init_client_trainer_state=self.init_state(),
        rng_seq=self._federated_algorithm._rng_seq)

    self.assertEqual(state.num_examples, 20)

  def test_train_single_client_iterator(self):
    state = client_trainer.train_single_client(
        dataset=self._client_dataset.as_numpy_iterator(),
        client_trainer=self._trainer,
        init_client_trainer_state=self.init_state(),
        rng_seq=self._federated_algorithm._rng_seq)

    self.assertEqual(state.num_examples, 20)

  @parameterized.named_parameters(
      {
          'testcase_name': 'sequential',
          'disable_parallel': 'true',
          'expected_num_examples': 20,
      },
      {
          'testcase_name': 'parallel',
          'disable_parallel': 'false',
          'expected_num_examples': 18,  # (20 // 3) * 3
      },
      {
          'testcase_name': 'auto',
          'disable_parallel': 'auto',
          'expected_num_examples': 18,  # (20 // 3) * 3
      })
  def test_train_multiple_clients(self, disable_parallel,
                                  expected_num_examples):
    with flagsaver.flagsaver(
        fedjax_experimental_disable_parallel=disable_parallel):
      state = self.init_state()
      states = client_trainer.train_multiple_clients(
          federated_data=self._federated_data,
          client_ids=self._federated_data.client_ids,
          client_trainer=self._trainer,
          init_client_trainer_state=state,
          rng_seq=self._federated_algorithm._rng_seq,
          client_data_hparams=self._client_data_hparams)
      states = list(states)

      self.assertLen(states, 5)
      for s in states:
        self.assertEqual(s.num_examples, expected_num_examples)


class ControlVariateTrainerTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._federated_algorithm = test_util.MockFederatedAlgorithm(
        num_clients=5, num_examples=20)
    self._federated_data = self._federated_algorithm.federated_data
    self._client_data_hparams = dataset_util.ClientDataHParams(batch_size=3)
    self._client_dataset = self._federated_data.create_tf_dataset_for_client(
        self._federated_data.client_ids[0]).batch(10)
    self._trainer = client_trainer.ControlVariateTrainer(
        self._federated_algorithm.model,
        optimizer.get_optimizer(
            optimizer.OptimizerName.MOMENTUM, learning_rate=0.2, momentum=0.9))

  def init_state(self):
    algo_state = self._federated_algorithm.init_state()
    opt_state = self._trainer._base_optimizer.init_fn(algo_state.params)
    control_variate = jax.tree_map(jnp.ones_like, algo_state.params)
    return self._trainer.init_state(algo_state.params, opt_state,
                                    control_variate)

  def test_one_step(self):
    init_state = self.init_state()
    batch = next(self._client_dataset.as_numpy_iterator())
    rng = next(self._federated_algorithm._rng_seq)

    state = self._trainer.one_step(init_state, batch, rng)

    self.assertEqual(state.num_examples, 10)
    jax.tree_multimap(self.assertAllEqual, state.control_variate,
                      init_state.control_variate)
    jax.tree_multimap(self.assertNotAllEqual, state.params, init_state.params)

  def test_loop(self):
    init_state = self.init_state()
    examples = zip(self._client_dataset.as_numpy_iterator(),
                   self._federated_algorithm._rng_seq)

    state = self._trainer.loop(init_state, examples)

    self.assertEqual(state.num_examples, 20)
    jax.tree_multimap(self.assertAllEqual, state.control_variate,
                      init_state.control_variate)
    jax.tree_multimap(self.assertNotAllEqual, state.params, init_state.params)

  @parameterized.named_parameters(
      {
          'testcase_name': 'sequential',
          'disable_parallel': 'true',
          'expected_num_examples': 20,
      },
      {
          'testcase_name': 'parallel',
          'disable_parallel': 'false',
          'expected_num_examples': 18,  # (20 // 3) * 3
      },
      {
          'testcase_name': 'auto',
          'disable_parallel': 'auto',
          'expected_num_examples': 18,  # (20 // 3) * 3
      })
  def test_train_multiple_clients(self, disable_parallel,
                                  expected_num_examples):
    with flagsaver.flagsaver(
        fedjax_experimental_disable_parallel=disable_parallel):
      init_state = self.init_state()
      states = client_trainer.train_multiple_clients(
          federated_data=self._federated_data,
          client_ids=self._federated_data.client_ids,
          client_trainer=self._trainer,
          init_client_trainer_state=init_state,
          rng_seq=self._federated_algorithm._rng_seq,
          client_data_hparams=self._client_data_hparams)
      states = list(states)

      self.assertLen(states, 5)
      for s in states:
        self.assertEqual(s.num_examples, expected_num_examples)
        jax.tree_multimap(self.assertAllEqual, s.control_variate,
                          init_state.control_variate)


if __name__ == '__main__':
  tf.test.main()
