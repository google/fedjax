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
"""Interface definitions for aggregators."""

from typing import Any, Callable, Iterable, Tuple
from fedjax.core import dataclasses
from fedjax.core import tree_util
from fedjax.core.typing import Params
from fedjax.core.typing import PRNGSequence

PyTree = Any
AggregatorState = PyTree


@dataclasses.dataclass
class Aggregator:
  """Interface for algorithms to aggregate.

  This interface defines aggregator algorithms that are used at each round.
  Aggregator state contains any round specific parameters (e.g. number of bits)
  that will be passed from round to round. This state is initialized by
  `init` and passed as input into and returned as output from `aggregate`.
  We strongly recommend using fedjax.dataclass to define state as this provides
  immutability, type hinting, and works by default with JAX transformations.

  Attributes:
    init: Returns initial state of aggregator.
    apply: Returns the new aggregator state and aggregated params.
  The expected usage of Aggregator is as follows:
  ```
   aggregator = mean_aggregator()
   state = aggregator.init()
   rng_seq = core.PRNGSequence(42)
   for i in range(num_rounds):
     params_and_weights = compute_client_outputs(i)
     state, aggregated_params = aggregator.apply(state, params_and_weights,
                                                 rng_seq)

  ```
  """
  init: Callable[[], AggregatorState]
  apply: Callable[
      [Iterable[Tuple[Params, float]], PRNGSequence, AggregatorState],
      Tuple[Params, AggregatorState]]


@dataclasses.dataclass
class MeanAggregatorState:
  """Mean aggregator is stateless."""


def mean_aggregator() -> Aggregator:
  """Builds (weighted) mean aggregator."""

  def init():
    return MeanAggregatorState()

  def apply(params_and_weights, rng, state):
    del rng
    return tree_util.tree_mean(params_and_weights), state

  return Aggregator(init, apply)
