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
"""Toy data and model for testing algorithms."""

import collections

from typing import List, Tuple

from fedjax.core import federated_algorithm
from fedjax.core import metrics
from fedjax.core import model
from fedjax.core.typing import FederatedData
import haiku as hk
import numpy as np
import optax
import tensorflow as tf
import tensorflow_federated as tff


def create_toy_data(num_clients: int, num_clusters: int, num_classes: int,
                    num_examples: int, seed: int) -> FederatedData:
  r"""Creates toy density estimation federated data.

  This is a synthetic dataset for density estimation. Let X=∅, Y=[d] where d
  is
  the number of classes. Let l be cross entropy loss and the number of users p.
  We create client distributions as a mixture with a uniform component, a
  cluster component, and an individual component. The details of the
  distributions are below.

  Let U be a uniform distribution over Y.  Let P_k be a point mass
  distribution with P_k(k) = 1.0 and for all y != k, P_k(y) = 0.
  For a client k, let D_k be a mixture given by

  D_k = 0.5 * P_{k%4} + 0.25 * U + 0.25 * P_{k%(d-4)}

  where a%b, is a modulo b. Roughly, U is the uniform component and is the same
  for all the clients, P_{k%4} is the cluster component and same for clients
  with same clusters, and P_{k%(d-4)} is the individual component per client.

  Args:
    num_clients: An int number of clients to generate.
    num_clusters: An int number of clusters to generate.
    num_classes: An int number of output classes.
    num_examples: An int number of examples per client.
    seed: An int random seed for sampling.

  Returns:
    Federated data containing randomly generated examples of structure:
      x: A numpy array of ones of shape [1].
      y: A numpy array of shape [].
  """
  distributions = 0.5 * np.identity(num_clusters)
  zeros = np.zeros((num_clusters, num_classes - num_clusters))
  distributions = np.concatenate([distributions, zeros], axis=-1)
  distributions += 0.25 * np.ones((num_clusters, num_classes)) / num_classes

  random_state = np.random.RandomState(seed)
  ids = random_state.permutation(num_clients) % num_clusters
  ids = np.arange(num_clients) % num_clusters

  client_data = collections.OrderedDict()
  for i in range(num_clients):
    local_change = np.zeros((num_classes))
    local_change[num_clusters + (i % (num_classes - num_clusters - 1)) +
                 1] = 0.25
    distribution = distributions[ids[i], :] + local_change
    client_id = str(i)
    client_data[client_id] = random_state.choice(
        num_classes, num_examples, p=distribution)

  for client_id, client_points in client_data.items():
    client_data[client_id] = collections.OrderedDict(
        x=tf.expand_dims(np.ones_like(client_points), axis=-1),
        y=client_points,
    )

  return tff.test.FromTensorSlicesClientData(client_data)


def create_toy_model(num_classes: int,
                     num_hidden_layers: int = 0,
                     feature_dim: int = 1) -> model.Model:
  """Creates toy haiku model.

  Args:
    num_classes: Number of output classes.
    num_hidden_layers: Number of hidden layers.
    feature_dim: Input feature dimension.

  Returns:
    Toy haiku model.
  """

  def forward_pass(batch):
    network = hk.Sequential(
        [hk.Linear(2 * num_classes) for _ in range(num_hidden_layers)] +
        [hk.Linear(num_classes)])
    return network(batch['x'])

  transformed_forward_pass = hk.transform(forward_pass)
  sample_batch = collections.OrderedDict(
      x=np.zeros((1, feature_dim)), y=np.zeros((1,)))

  def loss(batch, preds):
    return metrics.cross_entropy_loss_fn(targets=batch['y'], preds=preds)

  def accuracy(batch, preds):
    return metrics.accuracy_fn(targets=batch['y'], preds=preds)

  metrics_fn_map = collections.OrderedDict(accuracy=accuracy)
  return model.create_model_from_haiku(
      transformed_forward_pass=transformed_forward_pass,
      sample_batch=sample_batch,
      loss_fn=loss,
      metrics_fn_map=metrics_fn_map)


def create_toy_example(num_clients: int = 10,
                       num_clusters: int = 2,
                       num_classes: int = 4,
                       num_examples: int = 5,
                       num_hidden_layers: int = 0,
                       seed: int = 1) -> Tuple[FederatedData, model.Model]:
  """Creates toy data and model.

  Args:
    num_clients: Number of clients to generate.
    num_clusters: Number of clusters to generate.
    num_classes: Number of output classes.
    num_examples: Number of examples per client.
    num_hidden_layers: Number of hidden layers.
    seed: Random seed for sampling.

  Returns:
    A tuple of randomly generated federated toy data and toy model.
  """
  toy_data = create_toy_data(num_clients, num_clusters, num_classes,
                             num_examples, seed)
  toy_model = create_toy_model(num_classes, num_hidden_layers, feature_dim=1)
  return toy_data, toy_model


MockState = collections.namedtuple('MockState',
                                   ['params', 'opt_state', 'count'])


def create_mock_state(seed: int = 0) -> MockState:
  """Returns example nested struct with model parameters and optimizer state."""

  def net_fn(x):
    network = hk.Sequential([
        hk.Flatten(),
        hk.Linear(62),
    ])
    return network(x)

  transformed = hk.transform(net_fn)
  rng_seq = hk.PRNGSequence(seed)
  params = transformed.init(next(rng_seq), np.zeros((1, 1)))
  opt_init, _ = optax.sgd(learning_rate=1.0, momentum=0.9)
  opt_state = opt_init(params)
  return MockState(params=params, opt_state=opt_state, count=0)


class MockFederatedAlgorithm(federated_algorithm.FederatedAlgorithm):
  """Mock federated algorithm implementation."""

  def __init__(self, *args, **kwargs):
    self._federated_data, self._model = create_toy_example(*args, **kwargs)
    self._rng_seq = hk.PRNGSequence(0)

  @property
  def federated_data(self) -> FederatedData:
    return self._federated_data

  @property
  def model(self) -> model.Model:
    return self._model

  def init_state(self) -> MockState:
    """Returns initial state of algorithm."""
    params = self._model.init_params(next(self._rng_seq))
    opt_state = optax.sgd(learning_rate=1.0, momentum=0.9)[0](params)
    count = 0
    return MockState(params, opt_state, count)

  def run_round(self, state: MockState, client_ids: List[str]) -> MockState:
    """Runs one round of federated training."""
    count = state.count
    for client_id in client_ids:
      for _ in self._federated_data.create_tf_dataset_for_client(client_id):
        count += 1
    return MockState(
        params=state.params, opt_state=state.opt_state, count=count)
