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
"""Functions for working with tff.simulation.ClientData."""

import functools
from typing import Any, Iterable, Iterator, List, NamedTuple, Optional, Union

from fedjax.core.typing import FederatedData
import tensorflow as tf


def create_tf_dataset_for_clients(
    federated_data: FederatedData,
    client_ids: Optional[List[str]] = None) -> tf.data.Dataset:
  """Creates a single dataset by combining multiple input client datasets.

  Args:
    federated_data: Federated dataset.
    client_ids: Clients to combine. If None, combines all in client_data.

  Returns:
    A tf.data.Dataset of ordered mapping of combined examples.
  """
  client_ids = client_ids or federated_data.client_ids

  # create_tf_dataset_from_all_clients has fewer file reads than per client.
  if sorted(client_ids) == sorted(federated_data.client_ids):
    return federated_data.create_tf_dataset_from_all_clients()

  client_datasets = [
      federated_data.create_tf_dataset_for_client(c) for c in client_ids
  ]
  # Concatenate multiple client tf.data.Datasets into one tf.data.Dataset.
  return functools.reduce(lambda a, b: a.concatenate(b), client_datasets)


class ClientDataHParams(NamedTuple):
  """Hyperparameters for client data preparation.

  Attributes:
    batch_size: Batch size for training single or multiple clients.
    num_epochs: Number of epochs over data of single or multiple clients.
    drop_remainder: Whether to drop the last batch if it's < batch_size.
    shuffle_buffer_size: Maximum number of elements that will be buffered when
      prefetching. If 0, don't shuffle.
    num_batches: Maximum number of batches to include. Defaults to all.
  """
  batch_size: int = 1
  num_epochs: int = 1
  drop_remainder: bool = False
  shuffle_buffer_size: int = 0
  num_batches: int = -1


@tf.function
def preprocess_tf_dataset(dataset: tf.data.Dataset,
                          hparams: ClientDataHParams) -> tf.data.Dataset:
  """Preprocesses dataset according to the dataset hyperparmeters.

  Args:
    dataset: Dataset with a mapping element structure.
    hparams: Hyper parameters for dataset preparation.

  Returns:
    Preprocessed dataset.
  """
  dataset = dataset.repeat(hparams.num_epochs)
  if hparams.shuffle_buffer_size:
    dataset = dataset.shuffle(hparams.shuffle_buffer_size)
  dataset = (
      dataset.batch(hparams.batch_size,
                    drop_remainder=hparams.drop_remainder).prefetch(1))
  return dataset.take(hparams.num_batches)


DatasetOrIterable = Union[tf.data.Dataset, Iterable[Any]]


def iterate(dataset: DatasetOrIterable) -> Iterator[Any]:
  """Unified iteration over TF dataset or JAX supported types.

  This allows functions to take an argument from a range of types, while
  iterating over them in the same way. Currently the following are supported,

  - tf.data.Dataset: Iteration over a tf.data.Dataset directly produces numpy
  arrays.

  - Any other iterable: Such an iterable is directly iterated over.


  For example:

  def f(xs: DatasetOrIterable):
    total = 0
    for x in iterate(xs):
      total += x
    return total

  # Works with TF dataset.
  f(tf.data.Dataset.range(4))
  # Works with a numpy array list.
  f([np.ones((2, 2)), np.ones((1, 2)), np.ones((2, 1))])

  Args:
    dataset: Object to iterate over.

  Returns:
    An iterator.
  """
  if isinstance(dataset, tf.data.Dataset):
    return dataset.as_numpy_iterator()
  return iter(dataset)
