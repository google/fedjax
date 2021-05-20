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
"""Federated algorithm interface."""

from typing import Any, Callable, Mapping, Sequence, Tuple

from fedjax.core import client_datasets
from fedjax.core import dataclasses
from fedjax.core import federated_data
from fedjax.core.typing import PRNGKey
from fedjax.core.typing import PyTree


# Server state contains any round specific parameters (e.g. model parameters)
# that will be passed from round to round. This state must be serializable for
# checkpointing. We strongly recommend making this a PyTree.
ServerState = PyTree


@dataclasses.dataclass
class FederatedAlgorithm:
  """Container for all federated algorithms.

  FederatedAlgorithm defines the required methods that need to be implemented
  by all custom federated algorithms. Defining federated algorithms in this
  structure will allow implementations to work seamlessly with the convenience
  methods in the ``fedjax.training`` API, like checkpointing.

  Example toy implementation::

    # Federated algorithm that just counts total number of points across clients
    # across rounds.

    def count_federated_algorithm():

      def init(init_count):
        return {'count': init_count}

      def apply(state, clients):
        count = 0
        client_diagnostics = {}
        # Count sizes across clients.
        for client_id, client_dataset, _ in clients:
          # Summation across clients in one round.
          client_count = len(client_dataset)
          count += client_count
          client_diagnostics[client_id] = client_count
        # Summation across rounds.
        state = {'count': state['count'] + count}
        return state, client_diagnostics

      return FederatedAlgorithm(init, apply)

    rng = jax.random.PRNGKey(0)  # Unused.
    all_clients = [
        [
          (b'cid0', ClientDataset({'x': jnp.array([1, 2, 1, 2, 3, 4])}), rng),
          (b'cid1', ClientDataset({'x': jnp.array([1, 2, 3, 4, 5])}), rng),
          (b'cid2', ClientDataset({'x': jnp.array([1, 1, 2])}), rng),
        ],
        [
          (b'cid3', ClientDataset({'x': jnp.array([1, 2, 3, 4])}), rng),
          (b'cid4', ClientDataset({'x': jnp.array([1, 1, 2, 1, 2, 3])}), rng),
          (b'cid5', ClientDataset({'x': jnp.array([1, 2, 3, 4, 5, 6, 7])}), rng),
        ],
    ]
    algorithm = count_federated_algorithm()
    state = algorithm.init(0)
    for round_num in range(2):
      state, client_diagnostics = algorithm.apply(state, all_clients[round_num])
      print(round_num, state)
      print(round_num, client_diagnostics)
    # 0 {'count': 14}
    # 0 {b'cid0': 6, b'cid1': 5, b'cid2': 3}
    # 1 {'count': 31}
    # 1 {b'cid3': 4, b'cid4': 6, b'cid5': 7}

  Attributes:
    init: Initializes the ``ServerState``. Typically, the input to this method
      will be the initial model ``Params``. This should only be run once at the
      beginning of training.
    apply: Completes one round of federated training given an input
      ``ServerState`` and a sequence of tuples of client identifier, client
      dataset, and client rng. The output will be a new, updated ``ServerState``
      and accumulated per step results keyed by client identifier
      (e.g. train metrics).
  """
  init: Callable[..., ServerState]
  apply: Callable[[
      ServerState, Sequence[Tuple[federated_data.ClientId,
                                  client_datasets.ClientDataset, PRNGKey]]
  ], Tuple[ServerState, Mapping[federated_data.ClientId, Any]]]
