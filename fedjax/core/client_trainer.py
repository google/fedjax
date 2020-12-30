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
"""Functions and interfaces for single and multi client training."""

import abc
import functools
from typing import Any, Generic, Iterable, Iterator, List, NamedTuple, Optional, Tuple, TypeVar

from fedjax.core import dataset_util
from fedjax.core.model import Model
from fedjax.core.optimizer import Optimizer
from fedjax.core.typing import Batch
from fedjax.core.typing import FederatedData
from fedjax.core.typing import OptState
from fedjax.core.typing import Params
from fedjax.core.typing import PRNGKey
from fedjax.core.typing import PRNGSequence
from fedjax.core.typing import Updates
import jax
import tensorflow as tf

T = TypeVar('T')


class ClientTrainer(Generic[T], metaclass=abc.ABCMeta):
  """Interface for training a single client.

  This interface defines the core training components that interact with trainer
  state. Here, trainer state is a simple container holding only the necessary
  values for training e.g. model parameters and optimizer state.

  NOTE: Trainer state is distinct from the federated algorithm state. However,
  trainer state is typically derived from the federated algorithm state.
  """

  @abc.abstractmethod
  def init_state(self, *args, **kwargs) -> T:
    """Initializes the client trainer state."""

  @abc.abstractmethod
  def one_step(self, client_trainer_state: T, batch: Batch, rng: PRNGKey) -> T:
    """Runs one step of training on input batch and updates the trainer state.

    Args:
      client_trainer_state: Current client trainer state.
      batch: Batch of example numpy arrays of shape [client_batch_size, ...].
      rng: PRNGKey.

    Returns:
      Updated trainer state.
    """

  def loop(self, init_client_trainer_state: T,
           examples: Iterable[Tuple[Batch, PRNGKey]]) -> Any:
    """Runs training loop over multiple batches.

    For most usecases, this default implementation should work.

    Args:
      init_client_trainer_state: Initial client trainer state.
      examples: Sequence of batches and rngs, where batches are numpy arrays of
        shape [client_batch_size, ...].

    Returns:
      Output client trainer state that is typically just an updated version of
        the input `init_client_trainer_state`. However, output is flexible.
    """
    client_trainer_state = init_client_trainer_state
    for batch, rng in examples:
      client_trainer_state = self.one_step(client_trainer_state, batch, rng)
    return client_trainer_state


class DefaultClientTrainerState(NamedTuple):
  """State container for `DefaultClientTrainer`.

  Attributes:
    params: Pytree of model parameters.
    opt_state: Pytree of optimizer state.
    weight: Accumulated scalar weight over all batches.
  """
  params: Params
  opt_state: OptState
  weight: float = 0.


class DefaultClientTrainer(ClientTrainer):
  """Default single client trainer."""

  def __init__(self, model: Model, optimizer: Optimizer):
    super().__init__()
    self._model = model
    self._optimizer = optimizer

  def init_state(self,
                 params: Params,
                 opt_state: Optional[OptState] = None,
                 weight: float = 0.) -> DefaultClientTrainerState:
    if opt_state is None:
      opt_state = self._optimizer.init_fn(params)
    return DefaultClientTrainerState(
        params=params, opt_state=opt_state, weight=weight)

  @functools.partial(jax.jit, static_argnums=0)
  def one_step(self, client_trainer_state: DefaultClientTrainerState,
               batch: Batch, rng: PRNGKey) -> DefaultClientTrainerState:
    backward_pass_output = self._model.backward_pass(
        client_trainer_state.params, batch, rng)
    params_updates, opt_state = self._optimizer.update_fn(
        backward_pass_output.grads, client_trainer_state.opt_state)
    params = self._optimizer.apply_updates(client_trainer_state.params,
                                           params_updates)
    weight = client_trainer_state.weight + backward_pass_output.weight
    return DefaultClientTrainerState(
        params=params, opt_state=opt_state, weight=weight)


class ControlVariateTrainerState(NamedTuple):
  """State container for `ControlVariateTrainer`.

  Attributes:
    params: Pytree of model parameters.
    opt_state: Pytree of optimizer state.
    init_params: Pytree of initial model parameters.
    control_variate: Pytree of control variates that matches `params`.
    weight: Accumulated scalar weight over all batches.
  """
  params: Params
  opt_state: OptState
  init_params: Params
  control_variate: Updates
  weight: float = 0.


class ControlVariateTrainer(ClientTrainer):
  """Trainer with control variates."""

  def __init__(self, model: Model, base_optimizer: Optimizer):
    super().__init__()
    self._model = model
    self._base_optimizer = base_optimizer

  def init_state(self,
                 params: Params,
                 opt_state: OptState,
                 control_variate: Updates,
                 weight: float = 0.) -> ControlVariateTrainerState:
    return ControlVariateTrainerState(
        params=params,
        opt_state=opt_state,
        init_params=params,
        control_variate=control_variate,
        weight=weight)

  @functools.partial(jax.jit, static_argnums=0)
  def one_step(self, client_trainer_state: ControlVariateTrainerState,
               batch: Batch, rng: PRNGKey) -> ControlVariateTrainerState:
    client_control_variate = self._model.backward_pass(
        client_trainer_state.init_params, batch, rng).grads
    backward_pass_output = self._model.backward_pass(
        client_trainer_state.params, batch, rng)
    grads = backward_pass_output.grads
    adjusted_grads = jax.tree_multimap(lambda g, cc, c: g - cc + c, grads,
                                       client_control_variate,
                                       client_trainer_state.control_variate)
    updates, _ = self._base_optimizer.update_fn(adjusted_grads,
                                                client_trainer_state.opt_state)
    params = self._base_optimizer.apply_updates(client_trainer_state.params,
                                                updates)
    weight = client_trainer_state.weight + backward_pass_output.weight
    return ControlVariateTrainerState(
        params=params,
        opt_state=client_trainer_state.opt_state,
        init_params=client_trainer_state.init_params,
        control_variate=client_trainer_state.control_variate,
        weight=weight)


def train_single_client(dataset: tf.data.Dataset,
                        client_trainer: ClientTrainer[T],
                        init_client_trainer_state: T,
                        rng_seq: PRNGSequence) -> Any:
  """Trains model for a single client's dataset.

  Args:
    dataset: Dataset of examples [client_batch_size, d].
    client_trainer: ClientTrainer instance.
    init_client_trainer_state: Initial client trainer state. This will typically
      be derived from the federated algorithm state before calling
      `train_multiple_clients`.
    rng_seq: Random key generator.

  Returns:
    Output of client trainer that is typically just an updated version of the
      input `init_client_trainer_state`. However, output is flexible.
  """
  examples = zip(dataset.as_numpy_iterator(), rng_seq)
  client_trainer_state = client_trainer.loop(init_client_trainer_state,
                                             examples)
  return client_trainer_state


def train_multiple_clients(
    federated_data: FederatedData, client_ids: List[str],
    client_trainer: ClientTrainer[T], init_client_trainer_state: T,
    rng_seq: PRNGSequence,
    client_data_hparams: dataset_util.ClientDataHParams) -> Iterator[Any]:
  """Trains separate model for each client and records client updates.

  Args:
    federated_data: Federated data separated per client.
    client_ids: Ids of clients to train.
    client_trainer: ClientTrainer instance.
    init_client_trainer_state: Initial client trainer state. This will typically
      be derived from algorithm state before calling `train_multiple_clients`.
    rng_seq: Random key generator.
    client_data_hparams: Hyperparameters for client dataset preparation.

  Yields:
    Output of client trainer that is typically just an updated version of the
      input `init_client_trainer_state`. However, output is flexible.
  """
  for client_id in client_ids:
    client_dataset = federated_data.create_tf_dataset_for_client(client_id)
    client_dataset = dataset_util.preprocess_tf_dataset(client_dataset,
                                                        client_data_hparams)
    examples = zip(client_dataset.as_numpy_iterator(), rng_seq)
    client_trainer_state = client_trainer.loop(init_client_trainer_state,
                                               examples)
    yield client_trainer_state
