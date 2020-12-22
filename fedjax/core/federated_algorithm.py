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
"""Interface definitions for federated algorithms."""

import abc
from typing import Generic, List, TypeVar

from fedjax.core.model import Model
from fedjax.core.typing import FederatedData

T = TypeVar('T')


class FederatedAlgorithm(Generic[T], metaclass=abc.ABCMeta):
  """Interface for federated algorithms.

  This interface structures federated algorithms at the per round level.
  Algorithm state contains any round specific parameters (e.g. model parameters)
  that will be passed from round to round. This state is initialized by
  `init_state` and passed as input into and returned as output from `run_round`.
  We suggest defining state using `typing.NamedTuple` as this provides
  immutability, type hinting, and works by default with JAX transformations.

  Anything else that is static and doesn't update at each round can be defined
  in the `__init__`.

  The expected usage of FederatedAlgorithm is as follows:

  ```
  algorithm = FederatedAlgorithm()
  state = algorithm.init_state()
  for i in range(num_rounds):
    client_ids = sample_clients(i)
    state = algorithm.run_round(state, client_ids)
  ```
  """

  @abc.abstractproperty
  def federated_data(self) -> FederatedData:
    """Returns federated data used in federated training."""

  @abc.abstractproperty
  def model(self) -> Model:
    """Returns model used in federated training."""

  @abc.abstractmethod
  def init_state(self) -> T:
    """Returns initial state of algorithm."""

  @abc.abstractmethod
  def run_round(self, state: T, client_ids: List[str]) -> T:
    """Runs one round of federated training."""
