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
"""Structured flags commonly used in experiment binaries.

Structured flags are often used to construct complex structures via multiple
simple flags (e.g. an optimizer can be created by controlling learning rate and
other hyper parameters).
"""

import sys
from typing import Optional, Sequence

from absl import flags
from fedjax.core import client_datasets
from fedjax.core import optimizers
from fedjax.training import federated_experiment
from fedjax.training import tasks

FLAGS = flags.FLAGS


class NamedFlags:
  """A group of flags with an optional named prefix."""

  def __init__(self, name: Optional[str]):
    self._name = name
    if name is None:
      self._prefix = ''
      self._description = ''
    else:
      self._prefix = name + '_'
      self._description = f' for {name}'

  # Two special things:
  # - Add prefix/description when registering flags.
  # - Setting module_name so that these flags become visible for -help.

  def _enum(self, name: str, default: Optional[str], enum_values: Sequence[str],
            help_: str):
    flags.DEFINE_enum(
        self._prefix + name,
        default,
        enum_values,
        help_ + self._description,
        module_name=sys.argv[0])

  def _string(self, name: str, default: Optional[str], help_: str):
    flags.DEFINE_string(
        self._prefix + name,
        default,
        help_ + self._description,
        module_name=sys.argv[0])

  def _float(self, name: str, default: Optional[float], help_: str):
    flags.DEFINE_float(
        self._prefix + name,
        default,
        help_ + self._description,
        module_name=sys.argv[0])

  def _integer(self, name: str, default: Optional[int], help_: str):
    flags.DEFINE_integer(
        self._prefix + name,
        default,
        help_ + self._description,
        module_name=sys.argv[0])

  def _get_flag(self, flag):
    return getattr(FLAGS, self._prefix + flag)


class OptimizerFlags(NamedFlags):
  """Constructs a fedjax.Optimizer from flags."""

  SUPPORTED = ('sgd', 'momentum', 'adam', 'rmsprop', 'adagrad', 'yogi')

  def __init__(self,
               name: Optional[str] = None,
               default_optimizer: str = 'sgd'):
    super().__init__(name)
    self._enum('optimizer', default_optimizer, self.SUPPORTED, 'Optimizer')
    self._float('learning_rate', 0.005, 'Server step size')
    # Momentum parameters.
    self._float('momentum', 0.0, 'Momentum parameter')
    # Adam parameters
    self._float('adam_beta1', 0.9, 'Adam beta 1 parameter')
    self._float('adam_beta2', 0.99, 'Adam beta 2 parameter')
    self._float('adam_epsilon', 1e-3, 'Adam epsilon parameter')
    # RMSprop parameters.
    self._float('rmsprop_decay', 0.9, 'RMSProp decay parameter')
    self._float('rmsprop_epsilon', 1e-3, 'RMSprop epsilon parameter')
    # Adagrad parameters.
    self._float(
        'adagrad_epsilon', 1e-6,
        'Adagrad epsilon parameter that is added to second moment' +
        self._description)
    # Yogi parameters.
    self._float('yogi_beta1', 0.9, 'Yogi beta 1 parameter')
    self._float('yogi_beta2', 0.999, 'Yogi beta 2 parameter')
    self._float('yogi_epsilon', 1e-3, 'Yogi epsilon parameter')

  def get(self) -> optimizers.Optimizer:
    """Gets the specified optimizer."""
    optimizer_name = self._get_flag('optimizer')
    learning_rate = self._get_flag('learning_rate')
    if optimizer_name == 'sgd':
      return optimizers.sgd(learning_rate)
    elif optimizer_name == 'momentum':
      return optimizers.sgd(learning_rate, self._get_flag('momentum'))
    elif optimizer_name == 'adam':
      return optimizers.adam(learning_rate, self._get_flag('adam_beta1'),
                             self._get_flag('adam_beta2'),
                             self._get_flag('adam_epsilon'))
    elif optimizer_name == 'rmsprop':
      return optimizers.rmsprop(learning_rate, self._get_flag('rmsprop_decay'),
                                self._get_flag('rmsprop_epsilon'))
    elif optimizer_name == 'adagrad':
      return optimizers.adagrad(
          learning_rate, eps=self._get_flag('adagrad_epsilon'))
    elif optimizer_name == 'yogi':
      return optimizers.yogi(learning_rate, self._get_flag('yogi_beta1'),
                             self._get_flag('yogi_beta2'),
                             self._get_flag('yogi_epsilon'))
    else:
      raise ValueError(f'Unsupported optimizer {optimizer_name!r} from '
                       f'--{self._prefix}optimizer.')


class ShuffleRepeatBatchHParamsFlags(NamedFlags):
  """Constructs ShuffleRepeatBatchHParams from flags."""

  def __init__(self, name: Optional[str] = None, default_batch_size: int = 128):
    super().__init__(name)
    defaults = client_datasets.ShuffleRepeatBatchHParams(batch_size=-1)
    # TODO(wuke): Support other fields.
    self._integer('batch_size', default_batch_size, 'Batch size')
    self._integer('num_epochs', defaults.num_epochs, 'Number of epochs')
    self._integer('num_steps', defaults.num_steps, 'Number of steps')

  def get(self):
    return client_datasets.ShuffleRepeatBatchHParams(
        batch_size=self._get_flag('batch_size'),
        num_epochs=self._get_flag('num_epochs'),
        num_steps=self._get_flag('num_steps'))


class PaddedBatchHParamsFlags(NamedFlags):
  """Constructs PaddedBatchHParams from flags."""

  def __init__(self, name: Optional[str] = None, default_batch_size: int = 128):
    super().__init__(name)
    # TODO(wuke): Support other fields.
    self._integer('batch_size', default_batch_size, 'Batch size')

  def get(self):
    return client_datasets.PaddedBatchHParams(
        batch_size=self._get_flag('batch_size'))


class BatchHParamsFlags(NamedFlags):
  """Constructs BatchHParams from flags."""

  def __init__(self, name: Optional[str] = None, default_batch_size: int = 128):
    super().__init__(name)
    # TODO(wuke): Support other fields.
    self._integer('batch_size', default_batch_size, 'Batch size')

  def get(self):
    return client_datasets.BatchHParams(batch_size=self._get_flag('batch_size'))


class FederatedExperimentConfigFlags(NamedFlags):
  """Constructs FederatedExperimentConfig from flags."""

  def __init__(self, name: Optional[str] = None):
    super().__init__(name)
    defaults = federated_experiment.FederatedExperimentConfig(
        root_dir='', num_rounds=-1)
    self._string('root_dir', None, 'Root directory of experiment outputs')
    self._integer('num_rounds', None, 'Number of federated training rounds')
    self._integer(
        'checkpoint_frequency', defaults.checkpoint_frequency,
        'Checkpoint frequency in rounds' +
        '. If <= 0, no checkpointing is done.')
    self._integer('num_checkpoints_to_keep', defaults.num_checkpoints_to_keep,
                  'Maximum number of checkpoints to keep')
    self._integer(
        'eval_frequency', defaults.eval_frequency,
        'Evaluation frequency in rounds' + '. If <= 0, no evaluation is done.')

  def get(self):
    return federated_experiment.FederatedExperimentConfig(
        root_dir=self._get_flag('root_dir'),
        num_rounds=self._get_flag('num_rounds'),
        checkpoint_frequency=self._get_flag('checkpoint_frequency'),
        num_checkpoints_to_keep=self._get_flag('num_checkpoints_to_keep'),
        eval_frequency=self._get_flag('eval_frequency'))


class TaskFlags(NamedFlags):
  """Constructs a standard task tuple from flags."""

  def __init__(self, name: Optional[str] = None):
    super().__init__(name)
    self._enum('task', None, tasks.ALL_TASKS, 'Which task to run')
    self._string('data_mode', 'sqlite', 'Data loading mode')
    self._string('cache_dir', None,
                 'Cache directory when loading SQLite federated data')

  def get(self):
    return tasks.get_task(
        self._get_flag('task'), self._get_flag('data_mode'),
        self._get_flag('cache_dir'))
