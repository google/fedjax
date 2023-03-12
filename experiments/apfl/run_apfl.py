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
"""Top level experiment script for apfl.

Preset hyperparameters can be loaded via -flagfile, e.g.

python3 run_apfl.py -flagfile=apfl.EMNIST_DENSE.flags -root_dir=/tmp/apfl
"""

from functools import partial
from operator import itemgetter
import os.path
import time
from typing import Mapping, Sequence, Tuple

from absl import app
from absl import flags
from absl import logging

import fedjax
from fedjax import client_samplers
from fedjax import metrics
from fedjax.algorithms import apfl
from fedjax.training import checkpoint
from fedjax.training import logging as fedjax_logging
from fedjax.training import structured_flags

import jax
import jax.numpy as jnp

import tensorflow as tf


FLAGS = flags.FLAGS

flags.DEFINE_bool('use_parallel', False, 'Whether to use jax.pmap fedjax.for_each_client backend.')

task_flags = structured_flags.TaskFlags()

# Experiment configs.
experiment_config_flags = (structured_flags.FederatedExperimentConfigFlags())
eval_batch_flags = structured_flags.PaddedBatchHParamsFlags('eval')

# Random seeds.
flags.DEFINE_integer('params_seed', 1, 'Seed for initializing model parameters.')
flags.DEFINE_integer('train_sampler_seed', 2, 'Seed for the client sampler on the training set.')
flags.DEFINE_integer('test_sampler_seed', 3, 'Seed for the client sampler on the test set.')

# Training hyperparameters.
server_optimizer_flags = structured_flags.OptimizerFlags('server')
client_optimizer_flags = structured_flags.OptimizerFlags('client')
client_training_batch_flags = (structured_flags.ShuffleRepeatBatchHParamsFlags('client_training'))
flags.DEFINE_integer('num_clients_per_round', 340, 'Number of participating clients in each training round.')
flags.DEFINE_float('client_coefficient', 0.5, 'Initial interpolation coefficient.')


def aggregate_clients_eval(
  name: str,
  round_num: int,
  clients: Sequence[Tuple[fedjax.ClientId, Mapping[str, jnp.array]]],
  logger: fedjax_logging.Logger,
  model: fedjax.Model
) -> Mapping[str, jnp.array]:
  aggregate_stats = {k: metric.zero() for k, metric in model.eval_metrics.items()}
  for _, client_stats in clients:
    aggregate_stats = jax.tree_util.tree_map(
      lambda a, b: a.merge(b),
      aggregate_stats,
      client_stats,
      is_leaf=lambda v: isinstance(v, metrics.Stat))

  for metric_name, metric_value in aggregate_stats.items():
    logger.log(name, metric_name, metric_value.result(), round_num)


def full_clients_eval(
  clients: Sequence[Tuple[fedjax.ClientId, Mapping[str, jnp.array]]],
  path: str,
  model: fedjax.Model
) -> None:
  aggregate_stats = {k: metric.zero() for k, metric in model.eval_metrics.items()}
  with tf.io.gfile.GFile(path, 'w') as f:
    f.write('client_id' + '\t')
    f.write('\t'.join(model.eval_metrics.keys()) + '\n')

    for client_id, client_stats in clients:
      aggregate_stats = jax.tree_util.tree_map(
        lambda a, b: a.merge(b),
        aggregate_stats,
        client_stats,
        is_leaf=lambda v: isinstance(v, metrics.Stat))

      f.write(str(client_id) + '\t')
      f.write('\t'.join([str(v.result()) for v in client_stats.values()]) + '\n')

    f.write('aggregated' + '\t')
    f.write('\t'.join([str(v.result()) for v in aggregate_stats.values()]))


def main(argv: Sequence[str]) -> None:
  del argv

  config = experiment_config_flags.get()

  if FLAGS.use_parallel:
    fedjax.set_for_each_client_backend('pmap')

  if config.root_dir:
    tf.io.gfile.makedirs(config.root_dir)

  logger = fedjax_logging.Logger(config.root_dir)

  train_fd, test_fd, model = task_flags.get()

  algorithm = apfl.adaptive_personalized_federated_learning(
    grad_fn=fedjax.model_grad(model),
    server_optimizer=server_optimizer_flags.get(),
    client_optimizer=client_optimizer_flags.get(),
    client_batch_hparams=client_training_batch_flags.get(),
    client_coefficient=FLAGS.client_coefficient)

  eval_fn = apfl.eval_adaptive_personalized_federated_learning(
    model=model,
    client_batch_hparams=eval_batch_flags.get())

  train_client_sampler = client_samplers.UniformGetClientSampler(
    fd=train_fd,
    num_clients=FLAGS.num_clients_per_round,
    seed=FLAGS.train_sampler_seed)

  test_client_sampler = client_samplers.UniformGetClientSampler(
    fd=test_fd,
    num_clients=FLAGS.num_clients_per_round,
    seed=FLAGS.test_sampler_seed)

  aggregate_eval_fn = partial(
    aggregate_clients_eval,
    logger=logger,
    model=model
  )

  full_eval_fn = partial(
    full_clients_eval,
    path=os.path.join(config.root_dir, 'full_test_eval.tsv'),
    model=model
  )

  # Load checkpoint
  latest = checkpoint.load_latest_checkpoint(config.root_dir)
  if latest:
    state, last_round_num = latest
    start_round_num = last_round_num + 1
  else:
    state = algorithm.init(model.init(jax.random.PRNGKey(FLAGS.params_seed)))
    start_round_num = 1

  # Run training
  start = time.time()  
  for round_num in range(start_round_num, config.num_rounds + 1):
    # Get a random state and randomly sample clients.
    clients = train_client_sampler.sample()
    client_ids = [i[0] for i in clients]
    logging.info('round_num %d: client_ids = %s', round_num, client_ids)

    # Run one round of the algorithm, where bulk of the work happens.
    state, _ = algorithm.apply(state, clients)

    # Save checkpoint.
    should_save_checkpoint = config.checkpoint_frequency and (
      round_num == start_round_num or
      round_num % config.checkpoint_frequency == 0)
    if should_save_checkpoint:
      checkpoint.save_checkpoint(
        config.root_dir,
        state,
        round_num,
        config.num_checkpoints_to_keep)

    # Run evaluation.
    should_run_eval = config.eval_frequency and (
      round_num == start_round_num or
      round_num % config.eval_frequency == 0)
    if should_run_eval:
      start_periodic_eval = time.time()

      # Evaluate sampled train clients
      aggregate_eval_fn('fed_train_eval', round_num, eval_fn(state, map(itemgetter(0, 1), clients)))

      # Evaluate sampled test clients
      test_client_sampler.set_round_num(round_num)
      test_clients = test_client_sampler.sample()
      aggregate_eval_fn('fed_test_eval', round_num, eval_fn(state, map(itemgetter(0, 1), test_clients)))

      logger.log('.', 'periodic_eval_duration_sec', time.time() - start_periodic_eval, round_num)

    # Log the time it takes per round. Rough approximation since we're not
    # using DeviceArray.block_until_ready()
    logger.log(
      '.',
      'mean_round_duration_sec',
      (time.time() - start) / (round_num + 1 - start_round_num),
      round_num)

  # Block until previous work has finished.
  jnp.zeros([]).block_until_ready()

  # Logging overall time it took.
  num_rounds = config.num_rounds - start_round_num + 1
  mean_round_duration = ((time.time() - start) / num_rounds if num_rounds > 0 else 0)

  # Final evaluation.
  final_eval_start = time.time()
  full_eval_fn(eval_fn(state, test_fd.clients()))

  # DeviceArray.block_until_ready() isn't needed here since we write to file.
  final_eval_duration = time.time() - final_eval_start
  logging.info('mean_round_duration = %f sec.', mean_round_duration)
  logging.info('final_eval_duration = %f sec.', final_eval_duration)


if __name__ == '__main__':
  app.run(main)
