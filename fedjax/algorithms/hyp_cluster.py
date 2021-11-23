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
"""Federated hypothesis-based clustering implementation.

Three Approaches for Personalization with Applications to Federated Learning
    Yishay Mansour, Mehryar Mohri, Jae Ro, Ananda Theertha Suresh
    https://arxiv.org/abs/2002.10619
"""
import functools
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from fedjax.core import client_datasets
from fedjax.core import dataclasses
from fedjax.core import federated_algorithm
from fedjax.core import federated_data
from fedjax.core import for_each_client
from fedjax.core import models
from fedjax.core import optimizers
from fedjax.core import tree_util
from fedjax.core.typing import BatchExample
from fedjax.core.typing import OptState
from fedjax.core.typing import Params
from fedjax.core.typing import PRNGKey

import jax
import jax.numpy as jnp


@dataclasses.dataclass
class ServerState:
  """State of server for HypCluster passed between rounds.

  Attributes:
    cluster_params: A list of pytrees representing the server model parameters
      per cluster.
    opt_states: A list of pytrees representing the server optimizer state per
      cluster.
  """
  cluster_params: List[Params]
  opt_states: List[OptState]


def hyp_cluster(
    per_example_loss: Callable[[Params, BatchExample, PRNGKey], jnp.ndarray],
    client_optimizer: optimizers.Optimizer,
    server_optimizer: optimizers.Optimizer,
    maximization_batch_hparams: client_datasets.PaddedBatchHParams,
    expectation_batch_hparams: client_datasets.ShuffleRepeatBatchHParams,
    regularizer: Optional[Callable[[Params], jnp.ndarray]] = None
) -> federated_algorithm.FederatedAlgorithm:
  """Federated hypothesis-based clustering algorithm.

  Args:
    per_example_loss: A function from (params, batch, rng) to a vector of per
      example loss values.
    client_optimizer: Client side optimizer.
    server_optimizer: Server side optimizer.
    maximization_batch_hparams: Batching hyperparameters for the maximization
      step.
    expectation_batch_hparams: Batching hyperparameters for the expectation
      step.
    regularizer: Optional regularizer.

  Returns:
    A FederatedAlgorithm. A notable difference from other common
    FederatedAlgorithms such as fed_avg is that `init()` takes a list of
    `cluster_params`, which can be obtained using `random_init()`,
    `ModelKMeansInitializer`, or `kmeans_init()` in this module.
  """

  def init(cluster_params: List[Params]) -> ServerState:
    return ServerState(
        cluster_params,
        [server_optimizer.init(params) for params in cluster_params])

  # Creating these objects outside apply() can speed up repeated apply() calls.
  evaluator = models.AverageLossEvaluator(per_example_loss, regularizer)
  trainer = ClientDeltaTrainer(
      models.grad(per_example_loss, regularizer), client_optimizer)

  def apply(
      server_state: ServerState,
      clients: Sequence[Tuple[federated_data.ClientId,
                              client_datasets.ClientDataset, PRNGKey]]
  ) -> Tuple[ServerState, Mapping[federated_data.ClientId, Any]]:
    # Split RNGs for the 2 steps below.
    client_rngs = [jax.random.split(rng) for _, _, rng in clients]

    client_cluster_ids = maximization_step(
        evaluator=evaluator,
        cluster_params=server_state.cluster_params,
        clients=[(client_id, dataset, rng[0])
                 for (client_id, dataset, _), rng in zip(clients, client_rngs)],
        batch_hparams=maximization_batch_hparams)

    cluster_delta_params = expectation_step(
        trainer=trainer,
        cluster_params=server_state.cluster_params,
        client_cluster_ids=client_cluster_ids,
        clients=[(client_id, dataset, rng[1])
                 for (client_id, dataset, _), rng in zip(clients, client_rngs)],
        batch_hparams=expectation_batch_hparams)

    # Apply delta params for each cluster.
    cluster_params = []
    opt_states = []
    for delta_params, opt_state, params in zip(cluster_delta_params,
                                               server_state.opt_states,
                                               server_state.cluster_params):
      if delta_params is None:
        # No examples were observed for this cluster.
        next_opt_state, next_params = opt_state, params
      else:
        next_opt_state, next_params = server_optimizer.apply(
            delta_params, opt_state, params)
      cluster_params.append(next_params)
      opt_states.append(next_opt_state)

    # TODO(wuke): Other client diagnostics.
    client_diagnostics = {}
    for client_id, cluster_id in client_cluster_ids.items():
      client_diagnostics[client_id] = {'cluster_id': cluster_id}
    return ServerState(cluster_params, opt_states), client_diagnostics

  return federated_algorithm.FederatedAlgorithm(init, apply)


class _BaseClientTrainer:
  """Trains each client dataset, starting from either global params, or per client params."""

  def __init__(self, grad_fn: Callable[[Params, BatchExample, PRNGKey],
                                       jnp.ndarray],
               client_optimizer: optimizers.Optimizer, return_delta):
    # params can be passed in 2 ways:
    # -   As `shared_input`: All clients are trained starting from the same
    #     params.
    # -   As `client_input`: Each client is trained starting from per client
    #     params.
    def client_init(shared_input, client_input):
      if shared_input is not None:
        init_params = shared_input
        rng = client_input
      else:
        rng, init_params = client_input
      params = init_params
      opt_state = client_optimizer.init(params)
      return rng, params, opt_state, init_params

    def client_step(state, batch):
      rng, params, opt_state, init_params = state
      rng, use_rng = jax.random.split(rng)
      grads = grad_fn(params, batch, use_rng)
      opt_state, params = client_optimizer.apply(grads, opt_state, params)
      return rng, params, opt_state, init_params

    def client_final(shared_input, state):
      del shared_input
      _, params, _, init_params = state
      if return_delta:
        delta_params = jax.tree_util.tree_multimap(jnp.subtract, init_params,
                                                   params)
        return delta_params
      return params

    self._train_each_client = for_each_client.for_each_client(
        client_init, client_step, client_final)
    self._return_delta = return_delta

  def returns_delta(self):
    return self._return_delta

  def train_global_params(
      self, params: Params, clients: Iterable[Tuple[federated_data.ClientId,
                                                    Iterable[BatchExample],
                                                    PRNGKey]]
  ) -> Iterator[Tuple[federated_data.ClientId, Params]]:
    yield from self._train_each_client(shared_input=params, clients=clients)

  def train_per_client_params(
      self, clients: Iterable[Tuple[federated_data.ClientId,
                                    Iterable[BatchExample], PRNGKey, Params]]
  ) -> Iterator[Tuple[federated_data.ClientId, Params]]:
    yield from self._train_each_client(
        shared_input=None,
        clients=[(client_id, batches, (rng, params))
                 for client_id, batches, rng, params in clients])


class ClientDeltaTrainer(_BaseClientTrainer):
  """Trains each client dataset, starting from either global params, or per client params.

  The result values are delta params (i.e. initial params - final params).
  """

  def __init__(self, grad_fn: Callable[[Params, BatchExample, PRNGKey],
                                       jnp.ndarray],
               client_optimizer: optimizers.Optimizer):
    super().__init__(grad_fn, client_optimizer, return_delta=True)


class ClientParamsTrainer(_BaseClientTrainer):
  """Trains each client dataset, starting from either global params, or per client params.

  The result values are final params.
  """

  def __init__(self, grad_fn: Callable[[Params, BatchExample, PRNGKey],
                                       jnp.ndarray],
               client_optimizer: optimizers.Optimizer):
    super().__init__(grad_fn, client_optimizer, return_delta=False)


def maximization_step(
    evaluator: models.AverageLossEvaluator, cluster_params: List[Params],
    clients: Sequence[Tuple[federated_data.ClientId,
                            client_datasets.ClientDataset, PRNGKey]],
    batch_hparams: client_datasets.PaddedBatchHParams
) -> Dict[federated_data.ClientId, int]:
  """Evaluates and assigns each client to the cluster with the lowest average loss."""
  cluster_losses = _cluster_losses(
      evaluator=evaluator,
      cluster_params=cluster_params,
      clients=clients,
      batch_hparams=batch_hparams)
  return jax.device_get(_cluster_assignment(cluster_losses))


def _cluster_losses(
    evaluator: models.AverageLossEvaluator, cluster_params: List[Params],
    clients: Sequence[Tuple[federated_data.ClientId,
                            client_datasets.ClientDataset, PRNGKey]],
    batch_hparams: client_datasets.PaddedBatchHParams
) -> Dict[federated_data.ClientId, List[jnp.ndarray]]:
  """Evaluates the average loss for each client and cluster."""
  cluster_losses = {client_id: [] for client_id, _, _ in clients}
  client_rngs = [
      jax.random.split(rng, len(cluster_params)) for _, _, rng in clients
  ]
  for i, params in enumerate(cluster_params):
    for client_id, average_loss in evaluator.evaluate_global_params(
        params,
        [(client_id, dataset.padded_batch(batch_hparams), rng[i])
         for (client_id, dataset, _), rng in zip(clients, client_rngs)]):
      cluster_losses[client_id].append(average_loss)
  return cluster_losses


@jax.jit
def _cluster_assignment(cluster_losses):
  """Assigns each client the cluster with the lowest loss."""
  return {
      client_id: jnp.argmin(jnp.stack(losses))
      for client_id, losses in cluster_losses.items()
  }


def expectation_step(trainer: ClientDeltaTrainer, cluster_params: List[Params],
                     client_cluster_ids: Dict[federated_data.ClientId, int],
                     clients: Sequence[Tuple[federated_data.ClientId,
                                             client_datasets.ClientDataset,
                                             PRNGKey]],
                     batch_hparams: client_datasets.ShuffleRepeatBatchHParams):
  """Updates each cluster's params using average delta from clients in this cluster."""
  # Train each client starting from its corresponding cluster params, and
  # calculate weighted average of delta params within each cluster.
  num_examples = {client_id: len(dataset) for client_id, dataset, _ in clients}
  cluster_delta_params_sum = [
      jax.tree_util.tree_map(jnp.zeros_like, params)
      for params in cluster_params
  ]
  cluster_num_examples_sum = [0 for _ in cluster_params]
  for client_id, delta_params in trainer.train_per_client_params([
      (client_id, dataset.shuffle_repeat_batch(batch_hparams), rng,
       cluster_params[client_cluster_ids[client_id]])
      for client_id, dataset, rng in clients
  ]):
    cluster_id = client_cluster_ids[client_id]
    cluster_delta_params_sum[cluster_id] = tree_util.tree_add(
        cluster_delta_params_sum[cluster_id],
        tree_util.tree_weight(delta_params, num_examples[client_id]))
    cluster_num_examples_sum[cluster_id] += num_examples[client_id]
  # Weighted average delta params, or None if no examples were seen for this
  # cluster.
  cluster_delta_params = []
  for delta_params_sum, num_examples_sum in zip(cluster_delta_params_sum,
                                                cluster_num_examples_sum):
    if num_examples_sum > 0:
      cluster_delta_params.append(
          tree_util.tree_inverse_weight(delta_params_sum, num_examples_sum))
    else:
      cluster_delta_params.append(None)
  return cluster_delta_params


@functools.partial(jax.jit, static_argnums=(0, 1))
def random_init(num_clusters: int, init: Callable[[PRNGKey], Params],
                rng: PRNGKey) -> List[Params]:
  """Randomly initializes cluster params."""
  return [init(i) for i in jax.random.split(rng, num_clusters)]


class ModelKMeansInitializer:
  """Initializes cluster params for HypCluster using a k-means++ variant.

  This is a thin wrapper for initializing from a
  :class:`Model <fedjax.core.models.Model>` .
  See :func:`kmeans_init` for a more general version of the initializer, and
  details of the initialization algorithm.
  """

  def __init__(self,
               model: models.Model,
               client_optimizer: optimizers.Optimizer,
               regularizer: Optional[Callable[[Params], jnp.ndarray]] = None):
    self._model = model
    self._trainer = ClientParamsTrainer(
        models.model_grad(model, regularizer), client_optimizer)
    self._evaluator = models.AverageLossEvaluator(
        models.model_per_example_loss(model), regularizer)

  def cluster_params(
      self, num_clusters: int, rng: PRNGKey,
      clients: Sequence[Tuple[federated_data.ClientId,
                              client_datasets.ClientDataset, PRNGKey]],
      train_batch_hparams: client_datasets.ShuffleRepeatBatchHParams,
      eval_batch_hparams: client_datasets.PaddedBatchHParams) -> List[Params]:
    # Shared initial params.
    init_rng, center_rng = jax.random.split(rng)
    init_params = self._model.init(init_rng)
    return kmeans_init(
        num_clusters=num_clusters,
        init_params=init_params,
        clients=clients,
        trainer=self._trainer,
        train_batch_hparams=train_batch_hparams,
        evaluator=self._evaluator,
        eval_batch_hparams=eval_batch_hparams,
        rng=center_rng)


def kmeans_init(num_clusters: int, init_params: Params,
                clients: Sequence[Tuple[federated_data.ClientId,
                                        client_datasets.ClientDataset,
                                        PRNGKey]], trainer: ClientParamsTrainer,
                train_batch_hparams: client_datasets.ShuffleRepeatBatchHParams,
                evaluator: models.AverageLossEvaluator,
                eval_batch_hparams: client_datasets.PaddedBatchHParams,
                rng: PRNGKey) -> List[Params]:
  """Initializes cluster params for HypCluster using a k-means++ variant.

  See :class:`ModelKMeansInitializer` for a more convenient initializer from a
  :class:`Model <fedjax.core.models.Model>`.

  Given a set of input clients, we train parameters for each client. The
  initial cluster parameters are chosen out of this set of client parameters.
  At a high level, the cluster parameters are selected by choosing the client
  parameters that are the "furthest away" from each other (greatest difference
  in loss).

  Args:
    num_clusters: Desired number of cluster centers.
    init_params: Initial params for training each client.
    clients: Clients to train for generating candidate cluster center params.
    trainer: Client trainer for carry out client training.
    train_batch_hparams: Batching hyperparameters for client training.
    evaluator: Average loss evaluator for evaluator clients on chosen cluster
      center params.
    eval_batch_hparams: Batching hyperparameters for client evaluation.
    rng: RNG used for choosing the initial cluster center.

  Returns:
    Cluster center params.

  Raises:
    ValueError if some input arguments are invalid.
  """
  if num_clusters <= 0:
    raise ValueError(f'num_clusters should be > 0, got {num_clusters}')
  # Choose a client to be the initial center.
  initial_center_index = jax.random.choice(rng, len(clients))
  del rng  # Free up the name "rng" for subsequent loops.

  # Split client RNGs. For each client, we need 1 RNG for training,
  # (num_clusters - 1) for average loss evaluation.
  client_rngs = [jax.random.split(rng, num_clusters) for _, _, rng in clients]

  # Train each client from init_params.
  train_clients = [
      (client_id, dataset.shuffle_repeat_batch(train_batch_hparams), rng[0])
      for (client_id, dataset, _), rng in zip(clients, client_rngs)
  ]
  client_params = dict(trainer.train_global_params(init_params, train_clients))

  # TODO(wuke): Should clients already chosen be eliminated from subsequent
  # evaluations to prevent the same center from being added multiple times?

  # Choose cluster centers, starting with a randomly chosen initial center.
  cluster_centers = [client_params[sorted(client_params)[initial_center_index]]]
  # Repeatedly add the client with the worse best loss as the next center.
  best_losses = {client_id: jnp.inf for client_id in client_params}
  for i in range(1, num_clusters):
    # Evaluate all clients with the newly added center.
    eval_clients = [
        (client_id, dataset.padded_batch(eval_batch_hparams), rng[i])
        for (client_id, dataset, _), rng in zip(clients, client_rngs)
    ]
    new_losses = dict(
        evaluator.evaluate_global_params(cluster_centers[-1], eval_clients))
    # Update best loss for each client.
    best_losses = jax.tree_util.tree_multimap(min, best_losses,
                                              jax.device_get(new_losses))
    # Add a new center.
    worst_client = max(best_losses, key=lambda k: best_losses[k])
    cluster_centers.append(client_params[worst_client])

  return cluster_centers


class HypClusterEvaluator:
  """Evaluates cluster params on a Model."""

  def __init__(self,
               model: models.Model,
               regularizer: Optional[Callable[[Params], jnp.ndarray]] = None):
    """Initializes some reusable components.

    Because we need to make multiple passes over each client during evaluation,
    the number of clients that can be evaluated at once is limited. Therefore
    multiple calls to :meth:`~HypClusterEvaluator.evaluate_clients` are needed
    to evaluate a large federated dataset. We factor out some reusable
    components so that the same computation can be jit compiled.

    Args:
      model: Model being evaluated.
      regularizer: Optional regularizer.
    """
    self._maximization_step_evaluator = models.AverageLossEvaluator(
        models.model_per_example_loss(model), regularizer)
    self._model_evaluator = models.ModelEvaluator(model)

  def evaluate_clients(
      self, cluster_params: List[Params],
      train_clients: Sequence[Tuple[federated_data.ClientId,
                                    client_datasets.ClientDataset, PRNGKey]],
      test_clients: Sequence[Tuple[federated_data.ClientId,
                                   client_datasets.ClientDataset]],
      batch_hparams: client_datasets.PaddedBatchHParams
  ) -> Iterator[Tuple[federated_data.ClientId, Dict[str, jnp.ndarray]]]:
    """Evaluates each client on the cluster with best average loss."""
    cluster_client_ids = maximization_step(
        evaluator=self._maximization_step_evaluator,
        cluster_params=cluster_params,
        clients=train_clients,
        batch_hparams=batch_hparams)
    eval_clients = [(client_id, dataset.padded_batch(batch_hparams),
                     cluster_params[cluster_client_ids[client_id]])
                    for client_id, dataset in test_clients]
    yield from self._model_evaluator.evaluate_per_client_params(eval_clients)
