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
"""Sampling clients for participation in federated training rounds."""

import abc
from typing import Iterator, List, Tuple

from fedjax.core import client_datasets
from fedjax.core import federated_data
from fedjax.core.typing import PRNGKey

import jax
import numpy as np


class ClientSampler(abc.ABC):
  """Interface for client sampler.

  The output of `ClientSampler.sample` should be directly passed as input into
  `FederatedAlgorithm.apply`. In addition to sampling specific clients and their
  datasets, `ClientSampler` also handles generating a seed PRNGKey for each
  client.
  """

  @abc.abstractmethod
  def sample(
      self
  ) -> List[Tuple[federated_data.ClientId, client_datasets.ClientDataset,
                  PRNGKey]]:
    """Samples a subset of clients."""

  @abc.abstractmethod
  def set_round_num(self, round_num: int):
    """Sets round_num for the next sample round."""


class UniformShuffledClientSampler(ClientSampler):
  """Uniformly samples clients using `FederatedData.shuffled_clients`."""

  def __init__(
      self,
      shuffled_clients_iter: Iterator[Tuple[federated_data.ClientId,
                                            client_datasets.ClientDataset]],
      num_clients: int,
      start_round_num: int = 0):
    self._shuffled_clients_iter = shuffled_clients_iter
    self._num_clients = num_clients
    self._round_num = start_round_num
    # In case of restarting from latest checkpoint, we seek the shuffled clients
    # iterator to the previous `round_num - 1` round.
    for _ in range(self._round_num):
      for _ in range(self._num_clients):
        next(self._shuffled_clients_iter)

  def sample(
      self
  ) -> List[Tuple[federated_data.ClientId, client_datasets.ClientDataset,
                  PRNGKey]]:
    clients = []
    client_rngs = jax.random.split(
        jax.random.PRNGKey(self._round_num), self._num_clients)
    for i in range(self._num_clients):
      client_id, client_dataset = next(self._shuffled_clients_iter)
      clients.append((client_id, client_dataset, client_rngs[i]))
    self._round_num += 1
    return clients

  def set_round_num(self, round_num: int):
    raise NotImplementedError(
        "Use UniformGetClientSampler instead if set_round_num() is needed.")


class UniformGetClientSampler(ClientSampler):
  """Uniformly samples clients using `FederatedData.get_clients`."""

  def __init__(self,
               fd: federated_data.FederatedData,
               num_clients: int,
               seed: int,
               start_round_num: int = 0):
    self._federated_data = fd
    self._num_clients = num_clients
    self._seed = seed
    self._client_ids = list(self._federated_data.client_ids())
    self._round_num = start_round_num

  def sample(
      self
  ) -> List[Tuple[federated_data.ClientId, client_datasets.ClientDataset,
                  PRNGKey]]:
    clients = []
    random_state = get_pseudo_random_state(self._seed, self._round_num)
    # Explicitly specify dtype as np.object to prevent numpy from stripping
    # trailing zero bytes from client ids that resulted in lookup KeyErrors.
    client_ids = random_state.choice(
        np.array(self._client_ids, dtype=np.object),
        size=self._num_clients,
        replace=False)
    client_rngs = jax.random.split(
        jax.random.PRNGKey(self._round_num), self._num_clients)
    for i, (client_id, client_dataset) in enumerate(
        self._federated_data.get_clients(client_ids)):
      clients.append((client_id, client_dataset, client_rngs[i]))
    self._round_num += 1
    return clients

  def set_round_num(self, round_num: int):
    self._round_num = round_num


def get_pseudo_random_state(seed: int, round_num: int) -> np.random.RandomState:
  """Constructs a deterministic numpy random state."""
  #  Settings for a multiplicative linear congruential generator (aka Lehmer
  #  generator) suggested in 'Random Number Generators: Good Ones are Hard to
  # Find' by Park and Miller.
  mlcg_modulus = 2**(31) - 1
  mlcg_multiplier = 16807
  mlcg_start = np.random.RandomState(seed).randint(1, mlcg_modulus - 1)
  return np.random.RandomState(
      pow(mlcg_multiplier, round_num, mlcg_modulus) * mlcg_start % mlcg_modulus)
