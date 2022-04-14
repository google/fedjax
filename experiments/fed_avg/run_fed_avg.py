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
"""Top level experiment script for fed_avg.

Preset hyperparameters can be loaded via -flagfile, e.g.

python3 run_fed_avg.py -flagfile=fed_avg.EMNIST_CONV.flags -root_dir=/tmp/fed_avg
"""

from typing import Sequence

from absl import app
from absl import flags

import fedjax
import jax


FLAGS = flags.FLAGS
flags.DEFINE_bool('use_parallel', False,
                  'Whether to use jax.pmap fedjax.for_each_client backend.')

task_flags = fedjax.training.structured_flags.TaskFlags()

# Experiment configs.
experiment_config_flags = (
    fedjax.training.structured_flags.FederatedExperimentConfigFlags())
eval_batch_flags = fedjax.training.structured_flags.PaddedBatchHParamsFlags(
    'eval')

# Random seeds.
flags.DEFINE_integer('params_seed', 1,
                     'Seed for initializing model parameters.')
flags.DEFINE_integer('train_sampler_seed', 2,
                     'Seed for the client sampler on the training set.')
flags.DEFINE_integer('test_sampler_seed', 3,
                     'Seed for the client sampler on the test set.')

# Training hyperparameters.
server_optimizer_flags = fedjax.training.structured_flags.OptimizerFlags(
    'server')
client_optimizer_flags = fedjax.training.structured_flags.OptimizerFlags(
    'client')
client_training_batch_flags = (
    fedjax.training.structured_flags.ShuffleRepeatBatchHParamsFlags(
        'client_training'))
flags.DEFINE_integer('num_clients_per_round', 10,
                     'Number of participating clients in each training round.')


def main(argv: Sequence[str]) -> None:
  del argv
  if FLAGS.use_parallel:
    fedjax.set_for_each_client_backend('pmap')

  train_fd, test_fd, model = task_flags.get()
  run_full_periodic_eval = (
      FLAGS.task.startswith('EMNIST_') or
      FLAGS.task.startswith('SHAKESPEARE_') or
      FLAGS.task.startswith('CIFAR100_'))

  algorithm = fedjax.algorithms.fed_avg.federated_averaging(
      grad_fn=fedjax.model_grad(model),
      server_optimizer=server_optimizer_flags.get(),
      client_optimizer=client_optimizer_flags.get(),
      client_batch_hparams=client_training_batch_flags.get())

  config = experiment_config_flags.get()

  train_client_sampler = fedjax.client_samplers.UniformGetClientSampler(
      fd=train_fd,
      num_clients=FLAGS.num_clients_per_round,
      seed=FLAGS.train_sampler_seed)
  eval_batch_hparams = eval_batch_flags.get()
  periodic_eval_fn_map = {
      'fed_train_eval':
          fedjax.training.ModelTrainClientsEvaluationFn(model,
                                                        eval_batch_hparams),
      'fed_test_eval':
          fedjax.training.ModelSampleClientsEvaluationFn(
              fedjax.client_samplers.UniformGetClientSampler(
                  fd=test_fd,
                  num_clients=FLAGS.num_clients_per_round,
                  seed=FLAGS.test_sampler_seed), model, eval_batch_hparams)
  }
  final_eval_fn_map = {
      'full_test_eval':
          fedjax.training.ModelFullEvaluationFn(test_fd, model,
                                                eval_batch_hparams)
  }
  if run_full_periodic_eval:
    periodic_eval_fn_map.update(final_eval_fn_map)

  init_state = algorithm.init(model.init(jax.random.PRNGKey(FLAGS.params_seed)))
  fedjax.training.run_federated_experiment(
      algorithm=algorithm,
      init_state=init_state,
      client_sampler=train_client_sampler,
      config=config,
      periodic_eval_fn_map=periodic_eval_fn_map,
      final_eval_fn_map=final_eval_fn_map)


if __name__ == '__main__':
  app.run(main)
