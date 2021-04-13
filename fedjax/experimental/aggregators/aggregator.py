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

import abc
from typing import Iterable, Generic, NamedTuple, Tuple, TypeVar
from fedjax import core

T = TypeVar('T')
W = TypeVar('W')


class Aggregator(Generic[T], metaclass=abc.ABCMeta):
  """Interface for algorithms to aggregate.

  This interface structures aggregator algorithms that is used each round.
  Aggregate state contains any round specific parameters (e.g. number of bits)
  that will be passed from round to round. This state is initialized by
  `init_state` and passed as input into and returned as output from `aggregate`.
  We suggest defining state using `typing.NamedTuple` as this provides
  immutability, type hinting, and works by default with JAX transformations.

  Anything else that is static and doesn't update at each round can be defined
  in the `__init__`.

  The expected usage of Aggregator is as follows:

  ```
  aggregator = Aggregator()
  state = aggregator.init_state()
  rng_seq = core.PRNGSequence(42)
  for i in range(num_rounds):
    params_and_weights = compute_client_outputs(i)
    state, aggregated_params = aggregator.aggregate(state,
                                                    params_and_weights,
                                                    rng_seq)
  ```
  """

  @abc.abstractmethod
  def init_state(self) -> T:
    """Returns initial state of aggregator."""

  @abc.abstractmethod
  def aggregate(self, aggregator_state: T,
                params_and_weight: Iterable[Tuple[W, float]],
                rng_seq: core.PRNGSequence) -> Tuple[T, W]:
    """Returns the aggregated params and new state."""


class MeanAggregatorState(NamedTuple):
  """State of default aggregator passed between rounds.

  Attributes:
    total_weight: the number of samples
  """
  total_weight: float


class MeanAggregator(Aggregator):
  """Default aggregator that sums params according to weights."""

  def init_state(self):
    return MeanAggregatorState(total_weight=0.0)

  def aggregate(
      self, aggregator_state: MeanAggregatorState,
      params_and_weight: Iterable[Tuple[W, float]],
      rng_seq: core.PRNGSequence
  ) -> Tuple[MeanAggregatorState, W]:
    """Returns (weighted) mean of input trees and weights.

    Args:
      aggregator_state: State of the input.
      params_and_weight: Iterable of tuples of pytrees and associated weights.
      rng_seq: Random sequence.

    Returns:
      New state, (Weighted) mean of input trees and weights.
    """
    del rng_seq
    # TODO(theertha): avoid the need to copying the entire sequence to memory.
    params_and_weight = list(params_and_weight)
    weights = [weight for params, weight in params_and_weight]
    new_state = MeanAggregatorState(aggregator_state.total_weight +
                                    sum(weights))
    return core.tree_mean(params_and_weight), new_state
