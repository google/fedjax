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
"""Federated hypothesis-based clustering implementation.

Based on the paper:

Three Approaches for Personalization with Applications to Federated Learning
    Yishay Mansour, Mehryar Mohri, Jae Ro, Ananda Theertha Suresh
    https://arxiv.org/abs/2002.10619
"""

from typing import Iterator, List, Mapping, NamedTuple, Optional

from fedjax import core
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np


class HypClusterState(NamedTuple):
  """State of server for HypCluster passed between rounds.

  Attributes:
    cluster_params: A list of pytrees representing the server model parameters
      per cluster.
    cluster_server_opt_states: A list of pytrees representing the server
      optimizer state per cluster.
  """
  cluster_params: List[core.Params]
  cluster_server_opt_states: List[core.OptState]


class HypClusterHParams(NamedTuple):
  """Hyperparameters for federated hypothesis-based clustering.

  Attributes:
    train_data_hparams: Hyperparameters for training client data preparation.
    num_clusters: Number of clusters.
    init_cluster_client_ids: Optional list of client ids used to initialize
      cluster centers via k-means++ variant. If None, initialization is random.
  """
  train_data_hparams: core.ClientDataHParams
  num_clusters: int = 1
  init_cluster_client_ids: Optional[List[str]] = None


def _get_cluster_by_client_loss(
    federated_data: core.FederatedData,
    client_ids: List[str],
    model: core.Model,
    cluster_params: List[core.Params],
    data_hparams: Optional[core.ClientDataHParams] = None) -> np.ndarray:
  """Returns the loss per cluster and client pair.

  Args:
    federated_data: Federated data separated per client to be evaluated.
    client_ids: Ids of clients to be evaluated.
    model: Model implementation.
    cluster_params: List of pytree model parameters per cluster.
    data_hparams: Hyperparameters for dataset preparation.

  Returns:
    Numpy array of losses of shape [num_clusters, num_clients].
  """
  cluster_by_client_loss = []
  for params in cluster_params:
    client_metrics = core.evaluate_multiple_clients(
        federated_data=federated_data,
        client_ids=client_ids,
        model=model,
        params=params,
        client_data_hparams=data_hparams)
    client_loss = [m['loss'] if m else np.inf for m in client_metrics]
    cluster_by_client_loss.append(client_loss)
  # [num_clusters, num_clients].
  cluster_by_client_loss = np.array(cluster_by_client_loss)
  return cluster_by_client_loss


def maximization(
    federated_data: core.FederatedData,
    client_ids: List[str],
    model: core.Model,
    cluster_params: List[core.Params],
    data_hparams: Optional[core.ClientDataHParams] = None) -> List[List[str]]:
  """Finds the cluster that produces the lowest loss for each client.

  Args:
    federated_data: Federated data separated per client to be evaluated.
    client_ids: Ids of clients to be evaluated.
    model: Model implementation.
    cluster_params: List of pytree model parameters per cluster.
    data_hparams: Hyperparameters for dataset preparation.

  Returns:
    List of lists of client ids per cluster.
  """
  cluster_by_client_loss = _get_cluster_by_client_loss(federated_data,
                                                       client_ids, model,
                                                       cluster_params,
                                                       data_hparams)
  # [num_clients] index of best cluster per client.
  cluster_ids = np.argmin(cluster_by_client_loss, axis=0)

  cluster_client_ids = [[] for _ in cluster_params]
  for cluster_id, client_id in zip(cluster_ids, client_ids):
    cluster_client_ids[cluster_id].append(client_id)
  return cluster_client_ids


# TODO(b/162864512): Break up functions to jit compile.
class HypCluster(core.FederatedAlgorithm):
  """Federated hypothesis-based clustering algorithm."""

  def __init__(self, federated_data: core.FederatedData, model: core.Model,
               client_optimizer: core.Optimizer,
               server_optimizer: core.Optimizer, hparams: HypClusterHParams,
               rng_seq: core.PRNGSequence):
    """Initializes HypCluster algorithm.

    Args:
      federated_data: Federated data separated per client.
      model: Model implementation.
      client_optimizer: Client optimizer.
      server_optimizer: Server optimizer.
      hparams: Hyperparameters for federated hypothesis-based clustering.
      rng_seq: Iterator of JAX random keys.
    """
    self._federated_data = federated_data
    self._model = model
    self._client_optimizer = client_optimizer
    self._server_optimizer = server_optimizer
    self._hparams = hparams
    self._rng_seq = rng_seq
    self._client_trainer = core.DefaultClientTrainer(model, client_optimizer)

  def _kmeans_init(self) -> List[core.Params]:
    """Initializes cluster parameters for HypCluster using k-means++ variant.

    Given a set of input clients, we train parameters for each client. The
    initial cluster parameters are chosen out of this set of client parameters.
    At a high level, the cluster parameters are selected by choosing the client
    parameters that are the "furthest away" from each other (greatest difference
    in loss).

    Returns:
      Cluster center params.
    """
    init_params = self._model.init_params(next(self._rng_seq))
    client_ids = self._hparams.init_cluster_client_ids
    client_states = core.train_multiple_clients(
        federated_data=self.federated_data,
        client_ids=client_ids,
        client_trainer=self._client_trainer,
        init_client_trainer_state=self._client_trainer.init_state(
            params=init_params),
        rng_seq=self._rng_seq,
        client_data_hparams=self._hparams.train_data_hparams)
    params_by_client = [cs.params for cs in client_states]

    center_client = jrandom.choice(next(self._rng_seq), len(params_by_client))
    cluster_centers = [params_by_client[center_client]]

    for _ in range(1, self._hparams.num_clusters):
      # [num_centers, num_clients].
      cluster_by_client_loss = _get_cluster_by_client_loss(
          self.federated_data, client_ids, self.model, cluster_centers,
          self._hparams.train_data_hparams)
      # [num_clients].
      best_loss_per_client = np.min(cluster_by_client_loss, axis=0)
      worst_client = np.argmax(best_loss_per_client)
      cluster_centers.append(params_by_client[worst_client])

    return cluster_centers

  def _expectation(self, cluster_params: List[core.Params],
                   cluster_client_ids: List[List[str]]) -> List[core.Params]:
    """Trains each cluster's parameters on the clients in that cluster.

    Args:
      cluster_params: List of pytree model parameters per cluster.
      cluster_client_ids: List of lists of client ids per cluster.

    Returns:
      List of model parameter deltas per cluster.
    """
    cluster_delta_params = []
    for client_ids, params in zip(cluster_client_ids, cluster_params):
      if not client_ids:
        # Empty clusters shouldn't contribute to training since they have no
        # training data.
        cluster_delta_params.append(core.tree_util.tree_zeros_like(params))
        continue

      # Train cluster clients.
      client_states = core.train_multiple_clients(
          federated_data=self.federated_data,
          client_ids=client_ids,
          client_trainer=self._client_trainer,
          init_client_trainer_state=self._client_trainer.init_state(params),
          rng_seq=self._rng_seq,
          client_data_hparams=self._hparams.train_data_hparams)

      # Weighted average of param delta across clients.
      def select_delta_params_and_weight(client_state, p=params):
        delta_params = core.tree_multimap(lambda a, b: a - b, p,
                                          client_state.params)
        return delta_params, client_state.num_examples

      delta_params_and_weight = map(select_delta_params_and_weight,
                                    client_states)
      delta_params = core.tree_mean(delta_params_and_weight)
      cluster_delta_params.append(delta_params)
    return cluster_delta_params

  @property
  def federated_data(self) -> core.FederatedData:
    return self._federated_data

  @property
  def model(self) -> core.Model:
    return self._model

  def init_state(self) -> HypClusterState:
    if self._hparams.init_cluster_client_ids:
      cluster_params = self._kmeans_init()
    else:
      cluster_params = [
          self._model.init_params(next(self._rng_seq))
          for _ in range(self._hparams.num_clusters)
      ]

    cluster_server_opt_states = [
        self._server_optimizer.init_fn(p) for p in cluster_params
    ]
    return HypClusterState(
        cluster_params=cluster_params,
        cluster_server_opt_states=cluster_server_opt_states)

  def run_round(self, state: HypClusterState,
                client_ids: List[str]) -> HypClusterState:
    """Runs federated hypothesis-based clustering."""
    cluster_client_ids = maximization(self.federated_data, client_ids,
                                      self.model, state.cluster_params,
                                      self._hparams.train_data_hparams)
    cluster_delta_params = self._expectation(state.cluster_params,
                                             cluster_client_ids)

    cluster_params = state.cluster_params
    cluster_server_opt_states = state.cluster_server_opt_states
    for cluster_id, client_ids in enumerate(cluster_client_ids):
      if not client_ids:
        # Empty clusters shouldn't contribute to training since they have no
        # training data.
        continue
      updates, opt_state = self._server_optimizer.update_fn(
          cluster_delta_params[cluster_id],
          state.cluster_server_opt_states[cluster_id])
      cluster_server_opt_states[cluster_id] = opt_state
      cluster_params[cluster_id] = self._server_optimizer.apply_updates(
          state.cluster_params[cluster_id], updates)

    return HypClusterState(
        cluster_params=cluster_params,
        cluster_server_opt_states=cluster_server_opt_states)


def evaluate_multiple_clients_with_clusters(
    federated_data: core.FederatedData,
    client_ids: List[str],
    model: core.Model,
    cluster_params: List[core.Params],
    data_hparams: Optional[core.ClientDataHParams] = None
) -> Iterator[Mapping[str, jnp.ndarray]]:
  """Evaluates cluster parameters over input clients' datasets.

  Args:
    federated_data: Federated data separated per client to be evaluated.
    client_ids: Ids of clients to be evaluated.
    model: Model implementation.
    cluster_params: List of pytree model parameters per cluster.
    data_hparams: Hyperparameters for client dataset preparation.

  Yields:
    Ordered mapping of metric names to values per client.
  """
  cluster_client_ids = maximization(federated_data, client_ids, model,
                                    cluster_params, data_hparams)
  for client_ids_, params in zip(cluster_client_ids, cluster_params):
    client_metrics = core.evaluate_multiple_clients(
        federated_data=federated_data,
        client_ids=client_ids_,
        model=model,
        params=params,
        client_data_hparams=data_hparams)
    yield from client_metrics
