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
"""Tests for structured_flags."""

from absl.testing import absltest
from absl.testing import flagsaver
from fedjax.core import client_datasets
from fedjax.core import optimizers
from fedjax.training import federated_experiment
from fedjax.training import structured_flags


class OptimizerFlagsTest(absltest.TestCase):

  OPTIMIZER = structured_flags.OptimizerFlags()
  NAMED_OPTIMIZER = structured_flags.OptimizerFlags('client')

  def test_get(self):
    for optimizer in structured_flags.OptimizerFlags.SUPPORTED:
      with self.subTest('get ' + optimizer):
        with flagsaver.flagsaver(optimizer=optimizer):
          self.assertIsInstance(self.OPTIMIZER.get(), optimizers.Optimizer)
        with flagsaver.flagsaver(client_optimizer=optimizer):
          self.assertIsInstance(self.NAMED_OPTIMIZER.get(),
                                optimizers.Optimizer)
    with self.subTest('get foo'):
      with self.assertRaisesRegex(ValueError, 'from --optimizer'):
        with flagsaver.flagsaver(optimizer='foo'):
          self.OPTIMIZER.get()
      with self.assertRaisesRegex(ValueError, 'from --client_optimizer'):
        with flagsaver.flagsaver(client_optimizer='foo'):
          self.NAMED_OPTIMIZER.get()


class AllBatchHParamsFlagsTest(absltest.TestCase):

  SHUFFLE_BATCH_REPEAT = structured_flags.ShuffleRepeatBatchHParamsFlags(
      'shuffle_batch_repeat')
  PADDED_BATCH = structured_flags.PaddedBatchHParamsFlags('padded')
  BATCH = structured_flags.BatchHParamsFlags()

  def test_get(self):
    with self.subTest('default'):
      self.assertEqual(
          self.SHUFFLE_BATCH_REPEAT.get(),
          client_datasets.ShuffleRepeatBatchHParams(batch_size=128))
      self.assertEqual(self.PADDED_BATCH.get(),
                       client_datasets.PaddedBatchHParams(batch_size=128))
      self.assertEqual(self.BATCH.get(),
                       client_datasets.BatchHParams(batch_size=128))
    with self.subTest('custom'):
      with flagsaver.flagsaver(
          shuffle_batch_repeat_batch_size=12,
          shuffle_batch_repeat_num_epochs=21,
          padded_batch_size=34,
          batch_size=56):
        self.assertEqual(
            self.SHUFFLE_BATCH_REPEAT.get(),
            client_datasets.ShuffleRepeatBatchHParams(
                batch_size=12, num_epochs=21))
        self.assertEqual(self.PADDED_BATCH.get(),
                         client_datasets.PaddedBatchHParams(batch_size=34))
        self.assertEqual(self.BATCH.get(),
                         client_datasets.BatchHParams(batch_size=56))


class FederatedExperimentConfigTest(absltest.TestCase):

  FEDERATED_EXPERIMENT_CONFIG = structured_flags.FederatedExperimentConfigFlags(
  )

  @flagsaver.flagsaver(root_dir='foo', num_rounds=1234)
  def test_get(self):
    with self.subTest('default'):
      self.assertEqual(
          self.FEDERATED_EXPERIMENT_CONFIG.get(),
          federated_experiment.FederatedExperimentConfig(
              root_dir='foo', num_rounds=1234))
    with self.subTest('custom'):
      with flagsaver.flagsaver(
          root_dir='bar',
          num_rounds=567,
          checkpoint_frequency=2,
          num_checkpoints_to_keep=3,
          eval_frequency=4):
        self.assertEqual(
            self.FEDERATED_EXPERIMENT_CONFIG.get(),
            federated_experiment.FederatedExperimentConfig(
                root_dir='bar',
                num_rounds=567,
                checkpoint_frequency=2,
                num_checkpoints_to_keep=3,
                eval_frequency=4))


class TaskFlagsTest(absltest.TestCase):

  TASK = structured_flags.TaskFlags()

  def test_get(self):
    with self.subTest('default'):
      with self.assertRaisesRegex(ValueError, 'Unsupported task: None'):
        self.TASK.get()


if __name__ == '__main__':
  absltest.main()
