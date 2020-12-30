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
"""Training federated EMNIST model via federated averaging.

The model is a CNN with dropout, the client optimizer is vanilla SGD, and the
server optimizer is SGD with momentum.
"""

import collections

from absl import app
from absl import flags
from absl import logging

import fedjax
import simple_fed_avg
import jax
import tensorflow as tf

FLAGS = flags.FLAGS
# FederatedExperimentConfig flags.
flags.DEFINE_string('root_dir', '/tmp/emnist_simple_fed_avg',
                    'Root directory for experiment outputs (e.g. metrics).')
flags.DEFINE_integer('num_rounds', 1, 'Number of federated of training rounds.')
flags.DEFINE_integer('num_clients_per_round', 20,
                     'Number of clients per training round.')
flags.DEFINE_integer('sample_client_random_seed', None,
                     'Radom seed used to fix client sample per round.')
flags.DEFINE_integer(
    'checkpoint_frequency', 0,
    'Checkpoint frequency in rounds. If <= 0, no checkpointing is done.')
flags.DEFINE_integer('num_checkpoints_to_keep', 1,
                     'Maximum number of checkpoints to keep.')
flags.DEFINE_integer(
    'eval_frequency', 0,
    'Evaluation frequency in rounds. If <= 0, no evaluation is done.')

flags.DEFINE_string(
    'cache_dir', None,
    'Cache directory. If specified, files will be downloaded to disk. If '
    'unspecified, files are read directly over network.')
flags.DEFINE_bool('only_digits', False, 'Whether to use only digits or not.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('client_batch_size', 20, 'Client local batch size.')
flags.DEFINE_integer('client_num_epochs', 1, 'Client local number of epochs.')
flags.DEFINE_float('client_learning_rate', 0.02,
                   'Client optimizer learning rate.')
flags.DEFINE_float('server_learning_rate', 3.0,
                   'Server optimizer learning rate.')
flags.DEFINE_float('server_momentum', 0.9, 'Server optimizer momentum.')
flags.DEFINE_integer('eval_batch_size', 512, 'Evaluation batch size.')


def main(_):
  logging.info('Device count: %d, host count: %d, local device count: %d',
               jax.device_count(), jax.host_count(), jax.local_device_count())
  logging.info('Devices: %s', jax.devices())
  fedjax.training.set_tf_cpu_only()

  only_digits = FLAGS.only_digits
  train_federated_data, test_federated_data = fedjax.datasets.emnist.load_data(
      only_digits=only_digits, cache_dir=FLAGS.cache_dir)
  # Set tf.data.Dataset preprocessing functions to the tff.simulation.ClientData
  # that applies batching according to evaluation hyperparameters.
  train_federated_data_for_eval = train_federated_data.preprocess(
      lambda ds: ds.batch(FLAGS.eval_batch_size))
  test_federated_data_for_eval = test_federated_data.preprocess(
      lambda ds: ds.batch(FLAGS.eval_batch_size))

  model = fedjax.models.emnist.create_conv_model(only_digits=only_digits)
  hparams = simple_fed_avg.SimpleFedAvgHParams(
      train_data_hparams=fedjax.ClientDataHParams(
          batch_size=FLAGS.client_batch_size,
          num_epochs=FLAGS.client_num_epochs,
          shuffle_buffer_size=100,
      ))
  rng_seq = fedjax.PRNGSequence(FLAGS.seed)

  config = fedjax.training.FederatedExperimentConfig(
      root_dir=FLAGS.root_dir,
      num_rounds=FLAGS.num_rounds,
      num_clients_per_round=FLAGS.num_clients_per_round,
      sample_client_random_seed=FLAGS.sample_client_random_seed,
      checkpoint_frequency=FLAGS.checkpoint_frequency,
      num_checkpoints_to_keep=FLAGS.num_checkpoints_to_keep,
      eval_frequency=FLAGS.eval_frequency,
  )
  tf.io.gfile.makedirs(config.root_dir)

  federated_averaging = simple_fed_avg.SimpleFedAvg(
      federated_data=train_federated_data,
      model=model,
      client_optimizer=fedjax.get_optimizer(
          fedjax.OptimizerName.SGD, learning_rate=FLAGS.client_learning_rate),
      server_optimizer=fedjax.get_optimizer(
          fedjax.OptimizerName.MOMENTUM,
          learning_rate=FLAGS.server_learning_rate,
          momentum=FLAGS.server_momentum),
      hparams=hparams,
      rng_seq=rng_seq,
  )

  periodic_eval_fn_map = collections.OrderedDict(
      fed_train_eval=fedjax.training.ClientEvaluationFn(
          train_federated_data_for_eval, model, config),
      fed_test_eval=fedjax.training.ClientEvaluationFn(
          test_federated_data_for_eval, model, config),
      full_test_eval=fedjax.training.FullEvaluationFn(
          test_federated_data_for_eval, model))
  final_eval_fn_map = collections.OrderedDict(
      full_test_eval=fedjax.training.FullEvaluationFn(
          test_federated_data_for_eval, model))
  fedjax.training.run_federated_experiment(
      config=config,
      federated_algorithm=federated_averaging,
      periodic_eval_fn_map=periodic_eval_fn_map,
      final_eval_fn_map=final_eval_fn_map)


if __name__ == '__main__':
  app.run(main)
