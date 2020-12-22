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
"""EMNIST data loader."""

import collections
from typing import Optional, Tuple

from fedjax import core
import tensorflow as tf
import tensorflow_federated as tff

INIT_DOMAIN_WEIGHTS = (0.147, 0.853)  # HIGH_SCHOOL:CENSUS_FIELD.


def domain_id_fn(client_id: str) -> int:
  """Returns domain id for client id."""
  # These domain ids are based on NIST data source. For more details, see
  # https://s3.amazonaws.com/nist-srd/SD19/sd19_users_guide_edition_2.pdf.
  cid = int(client_id.split('_')[0][1:])
  if cid >= 2100 and cid <= 2599:
    return 0  # HIGH_SCHOOL.
  return 1  # CENSUS_FIELD.


def flip_and_expand(dataset: tf.data.Dataset) -> tf.data.Dataset:
  """Preprocesses dataset by expanding and inverting input pixel values.

  Args:
    dataset: Original source dataset.

  Returns:
    Dataset of ordered mapping of examples of the following structure:
      x: A numpy array of shape [28, 28, 1].
      y: A numpy array of shape [].
  """

  def map_fn(element):
    return collections.OrderedDict(
        x=tf.expand_dims(1.0 - element['pixels'], axis=-1),
        y=element['label'],
    )

  return dataset.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def load_data(
    only_digits: bool = True,
    cache_dir: Optional[str] = None
) -> Tuple[core.FederatedData, core.FederatedData]:
  """Loads EMNIST federated data with preprocessing.

  Args:
    only_digits: A bool whether to only include digits or not.
    cache_dir: Optional path to cache directory. If provided, files are
      downloaded over network to the specified cache directory.

  Returns:
    A tuple of federated data for train and test.
  """
  train_source, test_source = tff.simulation.datasets.emnist.load_data(
      only_digits=only_digits, cache_dir=cache_dir)
  return (train_source.preprocess(flip_and_expand),
          test_source.preprocess(flip_and_expand))
