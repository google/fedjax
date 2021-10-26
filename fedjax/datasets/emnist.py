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
"""Federated EMNIST."""

from typing import Optional, Tuple

from fedjax.core import client_datasets
from fedjax.core import federated_data
from fedjax.core import sqlite_federated_data
from fedjax.datasets import downloads
import numpy as np

SPLITS = ('train', 'test')


def cite():
  """Returns BibTeX citation for the dataset."""
  return """@inproceedings{cohen2017emnist,
  title={EMNIST: Extending MNIST to handwritten letters},
  author={Cohen, Gregory and Afshar, Saeed and Tapson, Jonathan and
Van Schaik, Andre},
  booktitle={2017 International Joint Conference on Neural Networks (IJCNN)},
  pages={2921--2926},
  year={2017},
  organization={IEEE}
}"""


def load_split(split: str,
               only_digits: bool = False,
               mode: str = 'sqlite',
               cache_dir: Optional[str] = None) -> federated_data.FederatedData:
  """Loads an unprocessed federated emnist split.

  Features:

  - pixels: [N, 28, 28] float32 image pixels.
  - label: [N] int32 classification label.

  Args:
    split: Name of the split. One of SPLITS.
    only_digits: Whether to only load the digits data.
    mode: 'sqlite'.
    cache_dir: Directory to cache files in 'sqlite' mode.

  Returns:
    FederatedData.
  """
  if split not in SPLITS:
    raise ValueError(f'Invalid split={split!r}')
  if cache_dir is not None and mode != 'sqlite':
    raise ValueError('Caching locally is only supported in "sqlite" mode')
  if only_digits:
    name = 'digitsonly_' + split
  else:
    name = split
  if mode == 'sqlite':
    path = downloads.maybe_download(
        f'https://storage.googleapis.com/gresearch/fedjax/emnist/federated_emnist_{name}.sqlite',
        cache_dir)
    return sqlite_federated_data.SQLiteFederatedData.new(path)
  else:
    raise ValueError(f'Unsupported mode={mode!r}')


def domain_id(client_id: federated_data.ClientId) -> int:
  """Returns domain id for client id.

  Domain ids are based on the NIST data source, where examples were collected
  from  two sources:
  Bethesda high school (HIGH_SCHOOL) and Census Bureau in Suitland (CENSUS).
  For more details, see the
  `NIST documentation <https://s3.amazonaws.com/nist-srd/SD19/sd19_users_guide_edition_2.pdf>`_.

  Args:
    client_id: Client id of the format
      ``[16-byte hex hash]:f[4-digit integer]_[2-digit integer]`` or
      ``f[4-digit integer]_[2-digit integer]``.

  Returns:
    Domain id that is 0 (HIGH_SCHOOL) or 1 (CENSUS).
  """
  # client ids are of the following format:
  # - sqlite: "[16-byte hex hash]:f[4-digit integer]_[2-digit integer]"
  #
  # These domain ids are based on NIST data source. For more details, see
  # https://s3.amazonaws.com/nist-srd/SD19/sd19_users_guide_edition_2.pdf.
  if len(client_id) == 25:
    cid = int(client_id[18:22])
  elif len(client_id) == 8:
    cid = int(client_id[1:5])
  else:
    raise ValueError(f'Invalid client_id: {client_id!r}')
  if 2100 <= cid and cid <= 2599:
    return 0  # HIGH_SCHOOL.
  return 1  # CENSUS.


def preprocess_client(
    client_id: federated_data.ClientId,
    examples: client_datasets.Examples) -> client_datasets.Examples:
  return {
      **examples, 'domain_id':
          np.full_like(examples['label'], domain_id(client_id))
  }


def preprocess_batch(
    examples: client_datasets.Examples) -> client_datasets.Examples:
  return {
      'x': 1 - examples['pixels'][..., np.newaxis],
      'y': examples['label'],
      'domain_id': examples['domain_id']
  }


def preprocess_split(
    fd: federated_data.FederatedData) -> federated_data.FederatedData:
  return (fd.preprocess_client(preprocess_client).preprocess_batch(
      preprocess_batch))


def load_data(
    only_digits: bool = False,
    mode: str = 'sqlite',
    cache_dir: Optional[str] = None
) -> Tuple[federated_data.FederatedData, federated_data.FederatedData]:
  """Loads processed EMNIST train and test splits.

  Features:

  - x: [N, 28, 28, 1] float32 flipped image pixels.
  - y: [N] int32 classification label.
  - domain_id: [N] int32 domain id (see :meth:`domain_id`).

  Args:
    only_digits: Whether to only load the digits data.
    mode: 'sqlite'.
    cache_dir: Directory to cache files in 'sqlite' mode.

  Returns:
    Train and test splits as FederatedData.
  """
  train = load_split(
      'train', only_digits=only_digits, mode=mode, cache_dir=cache_dir)
  test = load_split(
      'test', only_digits=only_digits, mode=mode, cache_dir=cache_dir)
  return preprocess_split(train), preprocess_split(test)
