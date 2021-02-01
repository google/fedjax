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
"""Lightweight wrappers for encapsulating JAX optimizers."""

import enum
from typing import Callable, Tuple

import dataclasses
from fedjax.core.typing import OptState
from fedjax.core.typing import Params
from fedjax.core.typing import Updates
import optax


@dataclasses.dataclass(frozen=True)
class Optimizer:
  """General optimizer interface.

  This is a simple container for the init_fn and update_fn functions returned by
  optax. Use apply_updates to apply the updates to the parameters.

  Attributes:
    init_fn: Initializes (possibly empty) sets of statistics (aka state).
    update_fn: Transforms a parameter update or gradient and updates the state.
    apply_updates: Apply the transformed gradients update to the parameters.
  """

  init_fn: Callable[[Params], OptState]
  update_fn: Callable[[Updates, OptState], Tuple[Updates, OptState]]
  apply_updates: Callable[[Params, Updates], Params] = optax.apply_updates


@enum.unique
class OptimizerName(enum.Enum):
  """Supported optimizers in fedjax."""
  # Standard SGD algorithm.
  SGD = 'SGD'
  # SGD algorithm with momentum.
  MOMENTUM = 'MOMENTUM'
  # Algorithm from arxiv.org/pdf/1412.6980.pdf
  ADAM = 'ADAM'
  # Algorithm from cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
  RMSPROP = 'RMSPROP'
  # Algorithm from https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
  ADAGRAD = 'ADAGRAD'


def get_optimizer(optimizer_name: OptimizerName,
                  learning_rate: float,
                  momentum: float = 0.0,
                  adam_beta1: float = 0.9,
                  adam_beta2: float = 0.999,
                  adam_epsilon: float = 1e-8,
                  rmsprop_decay: float = 0.9,
                  rmsprop_epsilon: float = 1e-8,
                  adagrad_init_accumulator: float = 0.1,
                  adagrad_epsilon: float = 1e-6) -> Optimizer:
  """Given parameters, returns the corresponding optimizer.

  Args:
    optimizer_name: One of SGD, MOMENTUM, ADAM, RMSPROP.
    learning_rate: Learning rate for all optimizers.
    momentum: Momentum parameter for MOMENTUM.
    adam_beta1: beta1 parameter for ADAM.
    adam_beta2: beta2 parameter for ADAM.
    adam_epsilon: epsilon parameter for ADAM.
    rmsprop_decay: decay parameter for RMSPROP.
    rmsprop_epsilon: epsilon parameter for RMSPROP.
    adagrad_init_accumulator: initial accumulator for ADAGRAD.
    adagrad_epsilon: epsilon parameter for ADAGRAD.

  Returns:
    Returns the Optimizer with the specified properties.

  Raises:
    ValueError: iff the optimizer names is not one of SGD, MOMENTUM, ADAM,
  RMSPROP, or Adagrad, raises errors.
  """
  if optimizer_name == OptimizerName.SGD:
    return Optimizer(*optax.sgd(learning_rate))
  elif optimizer_name == OptimizerName.MOMENTUM:
    return Optimizer(*optax.sgd(learning_rate, momentum))
  elif optimizer_name == OptimizerName.ADAM:
    return Optimizer(*optax.adam(
        learning_rate, b1=adam_beta1, b2=adam_beta2, eps=adam_epsilon))
  elif optimizer_name == OptimizerName.RMSPROP:
    return Optimizer(
        *optax.rmsprop(learning_rate, decay=rmsprop_decay, eps=rmsprop_epsilon))
  elif optimizer_name == OptimizerName.ADAGRAD:
    return Optimizer(*optax.adagrad(
        learning_rate,
        initial_accumulator_value=adagrad_init_accumulator,
        eps=adagrad_epsilon))
  else:
    raise ValueError(f'Unsupported optimizer_name {optimizer_name}.')
