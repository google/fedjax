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
"""FedJAX API."""

# Public core API.
from fedjax.core import client_samplers
from fedjax.core import metrics
from fedjax.core import optimizers
from fedjax.core import regularizers
from fedjax.core import serialization
from fedjax.core import tree_util

from fedjax.core.client_datasets import BatchHParams
from fedjax.core.client_datasets import BatchPreprocessor
from fedjax.core.client_datasets import buffered_shuffle_batch_client_datasets
from fedjax.core.client_datasets import ClientDataset
from fedjax.core.client_datasets import EXAMPLE_MASK_KEY
from fedjax.core.client_datasets import padded_batch_client_datasets
from fedjax.core.client_datasets import PaddedBatchHParams
from fedjax.core.client_datasets import ShuffleRepeatBatchHParams

from fedjax.core.dataclasses import dataclass

from fedjax.core.federated_algorithm import FederatedAlgorithm

from fedjax.core.federated_data import ClientId
from fedjax.core.federated_data import ClientPreprocessor
from fedjax.core.federated_data import FederatedData
from fedjax.core.federated_data import padded_batch_federated_data
from fedjax.core.federated_data import RepeatableIterator
from fedjax.core.federated_data import shuffle_repeat_batch_federated_data
from fedjax.core.federated_data import SubsetFederatedData
from fedjax.core.federated_data import FederatedDataBuilder

from fedjax.core.for_each_client import for_each_client
from fedjax.core.for_each_client import for_each_client_backend
from fedjax.core.for_each_client import get_for_each_client_backend
from fedjax.core.for_each_client import set_for_each_client_backend
from fedjax.core.for_each_client import ForEachClientBackend

from fedjax.core.models import AverageLossEvaluator
from fedjax.core.models import create_model_from_haiku
from fedjax.core.models import create_model_from_stax
from fedjax.core.models import evaluate_average_loss
from fedjax.core.models import evaluate_model
from fedjax.core.models import grad
from fedjax.core.models import Model
from fedjax.core.models import model_grad
from fedjax.core.models import model_per_example_loss
from fedjax.core.models import ModelEvaluator

from fedjax.core.sqlite_federated_data import SQLiteFederatedData
from fedjax.core.sqlite_federated_data import SQLiteFederatedDataBuilder

from fedjax.core.in_memory_federated_data import InMemoryFederatedData

from fedjax.core.optimizers import Optimizer
from fedjax.core.typing import BatchExample
from fedjax.core.typing import BatchPrediction
from fedjax.core.typing import OptState
from fedjax.core.typing import Params
from fedjax.core.typing import PRNGKey
from fedjax.core.typing import PyTree
from fedjax.core.typing import SingleExample
from fedjax.core.typing import SinglePrediction

# Subpackages available by default.
from fedjax import aggregators
from fedjax import algorithms
from fedjax import datasets
from fedjax import models
from fedjax import training

# Please do not use symbols in `core` as they are not part of the public API.
try:
  del core  # pylint: disable=undefined-variable
except NameError:
  pass
