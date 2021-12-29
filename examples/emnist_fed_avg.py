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
"""Training federated EMNIST via federated averaging.

- The model is a CNN with dropout
- The client optimizer is SGD
- The server optimizer is Adam.

Hyperparameters match those used in https://arxiv.org/abs/2003.00295.
"""

from absl import app

import fedjax
import fed_avg

import jax
import jax.numpy as jnp


def main(_):
  # Load train and test federated data for EMNIST.
  train_fd, test_fd = fedjax.datasets.emnist.load_data(only_digits=False)

  # Create CNN model with dropout.
  model = fedjax.models.emnist.create_conv_model(only_digits=False)

  # Scalar loss function with model parameters, batch of examples, and seed
  # PRNGKey as input.
  def loss(params, batch, rng):
    # `rng` used with `apply_for_train` to apply dropout during training.
    preds = model.apply_for_train(params, batch, rng)
    # Per example loss of shape [batch_size].
    example_loss = model.train_loss(batch, preds)
    return jnp.mean(example_loss)

  # Gradient function of `loss` w.r.t. to model `params` (jitted for speed).
  grad_fn = jax.jit(jax.grad(loss))

  # Create federated averaging algorithm.
  client_optimizer = fedjax.optimizers.sgd(learning_rate=10**(-1.5))
  server_optimizer = fedjax.optimizers.adam(
      learning_rate=10**(-2.5), b1=0.9, b2=0.999, eps=10**(-4))
  # Hyperparameters for client local traing dataset preparation.
  client_batch_hparams = fedjax.ShuffleRepeatBatchHParams(batch_size=20)
  algorithm = fed_avg.federated_averaging(grad_fn, client_optimizer,
                                          server_optimizer,
                                          client_batch_hparams)

  # Initialize model parameters and algorithm server state.
  init_params = model.init(jax.random.PRNGKey(17))
  server_state = algorithm.init(init_params)

  # Train and eval loop.
  train_client_sampler = fedjax.client_samplers.UniformGetClientSampler(
      fd=train_fd, num_clients=10, seed=0)
  for round_num in range(1, 1501):
    # Sample 10 clients per round without replacement for training.
    clients = train_client_sampler.sample()
    # Run one round of training on sampled clients.
    server_state, client_diagnostics = algorithm.apply(server_state, clients)
    print(f'[round {round_num}]')
    # Optionally print client diagnostics if curious about each client's model
    # update's l2 norm.
    # print(f'[round {round_num}] client_diagnostics={client_diagnostics}')

    if round_num % 10 == 0:
      # Periodically evaluate the trained server model parameters.
      # Read and combine clients' train and test datasets for evaluation.
      client_ids = [cid for cid, _, _ in clients]
      train_eval_datasets = [cds for _, cds in train_fd.get_clients(client_ids)]
      test_eval_datasets = [cds for _, cds in test_fd.get_clients(client_ids)]
      train_eval_batches = fedjax.padded_batch_client_datasets(
          train_eval_datasets, batch_size=256)
      test_eval_batches = fedjax.padded_batch_client_datasets(
          test_eval_datasets, batch_size=256)

      # Run evaluation metrics defined in `model.eval_metrics`.
      train_metrics = fedjax.evaluate_model(model, server_state.params,
                                            train_eval_batches)
      test_metrics = fedjax.evaluate_model(model, server_state.params,
                                           test_eval_batches)
      print(f'[round {round_num}] train_metrics={train_metrics}')
      print(f'[round {round_num}] test_metrics={test_metrics}')

  # Save final trained model parameters to file.
  fedjax.serialization.save_state(server_state.params, '/tmp/params')


if __name__ == '__main__':
  app.run(main)
