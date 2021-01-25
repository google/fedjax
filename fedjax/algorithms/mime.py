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
"""Mime implementation.

Based on the paper:

Mime: Mimicking Centralized Stochastic Algorithms in Federated Learning
    Sai Praneeth Karimireddy, Martin Jaggi, Satyen Kale, Mehryar Mohri,
    Sashank J. Reddi, Sebastian U. Stich, Ananda Theertha Suresh
    https://arxiv.org/abs/2008.03606
"""

from typing import List

from fedjax import core
from fedjax.algorithms import mime_lite


class Mime(core.FederatedAlgorithm):
  """Mime algorithm."""

  def __init__(self, federated_data: core.FederatedData, model: core.Model,
               base_optimizer: core.Optimizer,
               hparams: mime_lite.MimeLiteHParams, rng_seq: core.PRNGSequence):
    """Initializes MimeLite algorithm.

    Args:
      federated_data: Federated data separated per client.
      model: Model implementation.
      base_optimizer: Base centralized optimizer which we will try mimic.
      hparams: Hyperparameters for federated averaging.
      rng_seq: Iterator of JAX random keys.
    """
    self._federated_data = federated_data
    self._model = model
    self._base_optimizer = base_optimizer
    self._hparams = hparams
    self._rng_seq = rng_seq
    self._client_trainer = core.ControlVariateTrainer(model, base_optimizer)

  @property
  def federated_data(self) -> core.FederatedData:
    return self._federated_data

  @property
  def model(self) -> core.Model:
    return self._model

  def init_state(self) -> mime_lite.MimeLiteState:
    params = self._model.init_params(next(self._rng_seq))
    opt_state = self._base_optimizer.init_fn(params)
    return mime_lite.MimeLiteState(params, opt_state)

  def run_round(self, state: mime_lite.MimeLiteState,
                client_ids: List[str]) -> mime_lite.MimeLiteState:
    """Runs one round of Mime."""
    # Compute full-batch gradient at server params on train data.
    combined_dataset = core.preprocess_tf_dataset(
        core.create_tf_dataset_for_clients(self.federated_data, client_ids),
        self._hparams.combined_data_hparams)
    server_grads = mime_lite.compute_gradient(
        stream=combined_dataset.as_numpy_iterator(),
        params=state.params,
        model=self._model,
        rng_seq=self._rng_seq,
    )

    # Train on clients using custom ControlVariateTrainer.
    client_states = core.train_multiple_clients(
        federated_data=self.federated_data,
        client_ids=client_ids,
        client_trainer=self._client_trainer,
        init_client_trainer_state=self._client_trainer.init_state(
            params=state.params,
            opt_state=state.opt_state,
            control_variate=server_grads),
        rng_seq=self._rng_seq,
        client_data_hparams=self._hparams.train_data_hparams)

    # Weighted average of param delta across clients.
    def select_delta_params_and_weight(client_state):
      delta_params = core.tree_multimap(lambda a, b: a - b, state.params,
                                        client_state.params)
      return delta_params, client_state.num_examples

    delta_params_and_weight = map(select_delta_params_and_weight, client_states)
    delta_params = core.tree_mean(delta_params_and_weight)

    # Server params uses weighted average of client updates, scaled by the
    # server_learning_rate.
    params = core.tree_multimap(
        lambda p, q: p - self._hparams.server_learning_rate * q, state.params,
        delta_params)
    # Update server opt_state using base_optimimzer and server gradient.
    _, opt_state = self._base_optimizer.update_fn(server_grads, state.opt_state)
    return mime_lite.MimeLiteState(params, opt_state)
