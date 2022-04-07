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
"""Lightweight library for working with optimizers."""

from typing import Callable, List, Optional, Tuple, Union

from fedjax.core import dataclasses
from fedjax.core.typing import OptState
from fedjax.core.typing import Params

import haiku as hk
import jax
import optax

Grads = Params


@dataclasses.dataclass
class Optimizer:
  """Wraps different optimizer libraries in a common interface.

  Works with `optax <https://github.com/deepmind/optax>`_.

  The expected usage of Optimizer is as follows::

    # One step of SGD.
    params = {'w': jnp.array([1, 1, 1])}
    grads = {'w': jnp.array([2, 3, 4])}
    optimizer = fedjax.optimizers.sgd(learning_rate=0.1)
    opt_state = optimizer.init(params)
    opt_state, params = optimizer.apply(grads, opt_state, params)
    print(params)
    # {'w': DeviceArray([0.8, 0.7, 0.6], dtype=float32)}

  Attributes:
    init: Initializes (possibly empty) PyTree of statistics (optimizer state)
      given the input model parameters.
    apply: Transforms and applies the input gradients to update the optimizer
      state and model parameters.
  """
  init: Callable[[Params], OptState]
  apply: Callable[[Grads, OptState, Params], Tuple[OptState, Params]]


def create_optimizer_from_optax(opt: optax.GradientTransformation) -> Optimizer:
  """Creates optimizer from optax gradient transformation chain."""

  @jax.jit
  def apply(grads, opt_state, params):
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return opt_state, params

  return Optimizer(opt.init, apply)


def ignore_grads_haiku(optimizer: Optimizer,
                       non_trainable_names: List[Tuple[str, str]]) -> Optimizer:
  """Modifies ``optimizer`` to ignore gradients for ``non_trainable_names``.

  Non-trainable parameters will have their values set to ``None`` when passed as
  input into the Optimizer to prevent any updates.

  NOTE: This will only work with models implemented in haiku.

  Args:
    optimizer: Base Optimizer.
    non_trainable_names: List of tuples of haiku module names and names of
      given entries in the module data bundle (e.g. parameter name). This list
      of names will be used to select the non-trainable parameters.

  Returns:
    Optimizer that will ignore gradients for the non-trainable parameters.
  """
  non_trainable_names = tuple(non_trainable_names)

  def non_trainable_to_none(module_name, name, value):
    if (module_name, name) in non_trainable_names:
      return None
    return value

  def init(params):
    trainable_params = hk.data_structures.map(non_trainable_to_none, params)
    return optimizer.init(trainable_params)

  def apply(grads, opt_state, params):
    trainable_grads = hk.data_structures.map(non_trainable_to_none, grads)
    trainable_params = hk.data_structures.map(non_trainable_to_none, params)
    opt_state, trainable_params = optimizer.apply(trainable_grads, opt_state,
                                                  trainable_params)
    # Set the non-trainable params from None back to their original values.
    trainable_params = hk.data_structures.to_mutable_dict(trainable_params)
    for module_name, name in non_trainable_names:
      trainable_params[module_name][name] = params[module_name][name]
    return opt_state, hk.data_structures.to_immutable_dict(trainable_params)

  return Optimizer(init, apply)


# Convenient aliases for `optax` optimizers.
# Docstrings copied directly from `optax`.
ScalarOrSchedule = Union[float, optax.Schedule]


def adagrad(learning_rate: ScalarOrSchedule,
            initial_accumulator_value: float = 0.1,
            eps: float = 1e-6) -> Optimizer:
  """The Adagrad optimizer.

  Adagrad is an algorithm for gradient based optimisation that anneals the
  learning rate for each parameter during the course of training.

  WARNING: Adagrad's main limit is the monotonic accumulation of squared
  gradients in the denominator: since all terms are >0, the sum keeps growing
  during training and the learning rate eventually becomes vanishingly small.

  References:
    [Duchi et al, 2011](https://jmlr.org/papers/v12/duchi11a.html)

  Args:
    learning_rate: This is a fixed global scaling factor.
    initial_accumulator_value: Initialisation for the accumulator.
    eps: A small constant applied to denominator inside of the square root (as
      in RMSProp) to avoid dividing by zero when rescaling.

  Returns:
    The corresponding `Optimizer`.
  """
  return create_optimizer_from_optax(
      optax.adagrad(
          learning_rate=learning_rate,
          initial_accumulator_value=initial_accumulator_value,
          eps=eps))


def adam(learning_rate: ScalarOrSchedule,
         b1: float = 0.9,
         b2: float = 0.999,
         eps: float = 1e-8,
         eps_root: float = 0.0) -> Optimizer:
  """The classic Adam optimizer.

  Adam is an SGD variant with learning rate adaptation. The `learning_rate`
  used for each weight is computed from estimates of first- and second-order
  moments of the gradients (using suitable exponential moving averages).

  References:
    [Kingma et al, 2014](https://arxiv.org/abs/1412.6980)

  Args:
    learning_rate: This is a fixed global scaling factor.
    b1: The exponential decay rate to track the first moment of past gradients.
    b2: The exponential decay rate to track the second moment of past gradients.
    eps: A small constant applied to denominator outside of the square root (as
      in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: A small constant applied to denominator inside the square root (as
      in RMSProp), to avoid dividing by zero when rescaling. This is needed for
      example when computing (meta-)gradients through Adam.

  Returns:
    The corresponding `Optimizer`.
  """
  return create_optimizer_from_optax(
      optax.adam(
          learning_rate=learning_rate, b1=b1, b2=b2, eps=eps,
          eps_root=eps_root))


def rmsprop(learning_rate: ScalarOrSchedule,
            decay: float = 0.9,
            eps: float = 1e-8,
            initial_scale: float = 0.,
            centered: bool = False,
            momentum: Optional[float] = None,
            nesterov: bool = False) -> Optimizer:
  """A flexible RMSProp optimizer.

  RMSProp is an SGD variant with learning rate adaptation. The `learning_rate`
  used for each weight is scaled by a suitable estimate of the magnitude of the
  gradients on previous steps. Several variants of RMSProp can be found
  in the literature. This alias provides an easy to configure RMSProp
  optimizer that can be used to switch between several of these variants.

  References:
    [Tieleman and Hinton, 2012](www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
    [Graves, 2013](https://arxiv.org/abs/1308.0850)

  Args:
    learning_rate: This is a fixed global scaling factor.
    decay: The decay used to track the magnitude of previous gradients.
    eps: A small numerical constant to avoid dividing by zero when rescaling.
    initial_scale: Initialisation of accumulators tracking the magnitude of
      previous updates. PyTorch uses `0`, TF1 uses `1`. When reproducing results
      from a paper, verify the value used by the authors.
    centered: Whether the second moment or the variance of the past gradients is
      used to rescale the latest gradients.
    momentum: The `decay` rate used by the momentum term, when it is set to
      `None`, then momentum is not used at all.
    nesterov: Whether nesterov momentum is used.

  Returns:
    The corresponding `Optimizer`.
  """
  return create_optimizer_from_optax(
      optax.rmsprop(
          learning_rate=learning_rate,
          decay=decay,
          eps=eps,
          initial_scale=initial_scale,
          centered=centered,
          momentum=momentum,
          nesterov=nesterov))


def sgd(learning_rate: ScalarOrSchedule,
        momentum: Optional[float] = None,
        nesterov: bool = False) -> Optimizer:
  """A canonical Stochastic Gradient Descent optimizer.

  This implements stochastic gradient descent. It also includes support for
  momentum, and nesterov acceleration, as these are standard practice when
  using stochastic gradient descent to train deep neural networks.

  References:
    [Sutskever et al, 2013](http://proceedings.mlr.press/v28/sutskever13.pdf)

  Args:
    learning_rate: This is a fixed global scaling factor.
    momentum: The `decay` rate used by the momentum term, when it is set to
      `None`, then momentum is not used at all.
    nesterov: Whether nesterov momentum is used.

  Returns:
    The corresponding `Optimizer`.
  """
  return create_optimizer_from_optax(
      optax.sgd(
          learning_rate=learning_rate, momentum=momentum, nesterov=nesterov))


def yogi(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-3,
) -> Optimizer:
  """The Yogi optimizer.

  Yogi is an adaptive optimizer, which provides control in tuning the effective
  learning rate to prevent it from increasing. By doing so, it focuses on
  addressing the issues of convergence and generalisation in exponential moving
  average-based adaptive methods (such as Adam and RMSprop). Yogi is a
  modification of Adam and uses the same parameters.

  References:
    [Zaheer et al, 2020](http://www.sanjivk.com/yogi_nips2018.pdf)

  Args:
    learning_rate: this is a fixed global scaling factor.
    b1: the exponential decay rate to track the first moment of past gradients.
    b2: the exponential decay rate to track the second moment of past gradients.
    eps: a small constant applied to denominator outside of the square root (as
      in the Adam paper) to avoid dividing by zero when rescaling.

  Returns:
    The corresponding `Optimizer`.
  """
  return create_optimizer_from_optax(
      optax.yogi(learning_rate=learning_rate, b1=b1, b2=b2, eps=eps))
