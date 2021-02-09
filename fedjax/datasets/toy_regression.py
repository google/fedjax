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
"""Toy regression data loader."""

import collections
import functools
from typing import Tuple

from fedjax import core
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff


def domain_id_fn(client_id: str) -> int:
  return int(client_id.split('_')[-1])


def create_toy_regression_data(num_clients: int, num_domains: int,
                               num_points: int, client_domains: np.ndarray,
                               seed: int) -> core.FederatedData:
  r"""Creates toy regression federated data.

  This is a synthetic dataset for a simple regression example for domain
  agnostic learning. Let each domain i, be a set of points x_i_1, ..., x_i_m in
  R. Further, let the average of the points be the center of these points. We
  distribute these points on K clients randomly with the condition that each
  client k can only have points from a single domain i. The goal is to find a
  point that minimizes the maximum distance to all the domain centers.

  Args:
    num_clients: Number of clients to generate.
    num_domains: Number of domains to generate points for.
    num_points: Number of points to generate.
    client_domains: Domain id per client of shape [num_clients].
    seed: Random seed for sampling.

  Returns:
    Federated data containing randomly generated examples of structure:
      x: A numpy array of ones of shape [1].
      y: A numpy array of shape [].
  """
  random_state = np.random.RandomState(seed)

  # Generate points and assign them to domains.
  domains = random_state.randint(0, num_domains, size=[num_points])
  points = random_state.random_sample([num_points]) * 10
  for i in range(num_domains):
    points[domains == i] = points[domains == i] + (i * i)

  # Calculate domain averages and offset points by the center (min + max) / 2.
  avg_point = np.empty(num_domains)
  for i in range(0, num_domains):
    avg_point[i] = np.mean(points[domains == i])
  center = (np.max(avg_point) + np.min(avg_point)) / 2
  points = points - center

  # Group points by domain.
  domain_points = []
  for i in range(num_domains):
    domain_points.append(points[domains == i])
  clients = np.arange(num_clients)

  # Distribute points across clients.
  client_points = [[] for _ in range(num_clients)]
  for i in range(num_domains):
    points_i = domain_points[i]
    clients_i = clients[client_domains == i]
    points_clients = random_state.choice(clients_i, size=len(points_i))

    for j in clients_i:
      client_points[j] = points_i[points_clients == j]

  client_ids = [f'{c}_{d}' for c, d in zip(clients, client_domains)]
  client_data = collections.OrderedDict()
  for c, p in zip(client_ids, client_points):
    client_data[c] = collections.OrderedDict(
        x=tf.expand_dims(np.ones_like(p), axis=-1),
        y=p,
    )

  return tff.test.FromTensorSlicesClientData(client_data)


def load_data(num_clients: int, num_domains: int, num_points: int,
              seed: int) -> Tuple[core.FederatedData, core.FederatedData]:
  """Loads toy regression federated data.

  Args:
    num_clients: Number of clients to generate.
    num_domains: Number of domains to generate points for.
    num_points: Number of points to generate.
    seed: Random seed for sampling.

  Returns:
    Train and test toy regression federated data.
  """
  # Randomly assign each client to a domain.
  random_state = np.random.RandomState(seed)
  client_domains = random_state.randint(0, num_domains, size=[num_clients])
  create_data = functools.partial(
      create_toy_regression_data,
      num_clients=num_clients,
      num_domains=num_domains,
      num_points=num_points,
      client_domains=client_domains)
  train_data = create_data(seed=seed + 1)
  test_data = create_data(seed=seed + 2)
  return train_data, test_data
