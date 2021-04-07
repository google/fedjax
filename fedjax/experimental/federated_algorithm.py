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

from typing import Any, Callable, Iterable, Mapping, Tuple

from fedjax import core

PyTree = Any
# AlgorithmState contains any round specific parameters (e.g. model parameters)
# that will be passed from round to round. This state must be serializable for
# checkpointing. We strongly recommend making this a PyTree.
AlgorithmState = PyTree
# Client contains the client identifier and dataset as well as any per client
# information like persistent client state for the cross-silo setting.
Client = Tuple[str, Any]
# ClientDiagnostics contains the accumulated per step results keyed by client
# identifiers.
ClientDiagnostics = Mapping[str, Any]


@core.dataclass
class FederatedAlgorithm:
  """Container for all federated algorithms.

  `FederatedAlgorithm` defines the required methods that need to be implemented
  by all custom federated algorithms. Defining federated algorithms in this
  structure will allow implementations to work seamlessly with the convenience
  methods in the `fedjax.training` API, like checkpointing.

  Example toy implementation:

  ```python
  # Federated algorithm that just counts total number of points across clients
  # across rounds.

  def count_federated_algorithm():

    def init(init_count):
      return {'count': init_count}

    def run_one_round(state, clients):
      count = 0
      client_diagnostics = {}
      # Count sizes across clients.
      for client_id, client_dataset in clients:
        client_count = 0
        for d in client_dataset:
          client_count += sum(l.size for l in jax.tree_util.tree_leaves(d))
        # Summation across clients in one round.
        count += client_count
        client_diagnostics[client_id] = client_count
      # Summation across rounds.
      state = {'count': state['count'] + count}
      return state, client_diagnostics

    return FederatedAlgorithm(init, run_one_round)

  all_clients = [
      [
          ('cid0', [jnp.array([1, 2]), jnp.array([1, 2, 3, 4])]),
          ('cid1', [jnp.array([1, 2, 3, 4, 5])]),
          ('cid2', [jnp.array([1]), jnp.array([1, 2])]),
      ],
      [
          ('cid3', [jnp.array([1, 2, 3, 4])]),
          ('cid4', [jnp.array([1]), jnp.array([1, 2]), jnp.array([1, 2, 3])]),
          ('cid5', [jnp.array([1, 2, 3, 4, 5, 6, 7])]),
      ],
  ]
  algorithm = count_federated_algorithm()
  state = algorithm.init(0)
  for round_num in range(2):
    state, client_diagnostics = algorithm.run_one_round(state,
                                                        all_clients[round_num])
    print(round_num, state)
    print(round_num, client_diagnostics)
  # 0 {'count': 14}
  # 0 {'cid0': 6, 'cid1': 5, 'cid2': 3}
  # 1 {'count': 31}
  # 1 {'cid3': 4, 'cid4': 6, 'cid5': 7}
  ```

  Attributes:
    init: Initializes the `AlgorithmState`. Typically, the input to this method
      will be the initial model `Params`. This should only be run once at the
      beginning of training.
    run_one_round: Completes one round of federated training given an input
      `AlgorithmState` and a sequence of client identifiers to client datasets.
      The output will be a new, updated `AlgorithmState` along with any per
      client `DiagnosticInfo` (e.g. train metrics).
  """
  init: Callable[..., AlgorithmState]
  run_one_round: Callable[[AlgorithmState, Iterable[Client]],
                          Tuple[AlgorithmState, ClientDiagnostics]]
