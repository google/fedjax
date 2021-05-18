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
"""FedJAX core."""

# Test utilities.
from fedjax.legacy.core import test_util

# Client trainer.
from fedjax.legacy.core.client_trainer import ClientTrainer
from fedjax.legacy.core.client_trainer import ControlVariateTrainer
from fedjax.legacy.core.client_trainer import ControlVariateTrainerState
from fedjax.legacy.core.client_trainer import DefaultClientTrainer
from fedjax.legacy.core.client_trainer import DefaultClientTrainerState
from fedjax.legacy.core.client_trainer import train_multiple_clients
from fedjax.legacy.core.client_trainer import train_single_client
# Dataclass utils.
from fedjax.legacy.core.dataclasses import dataclass
# Dataset utilities.
from fedjax.legacy.core.dataset_util import ClientDataHParams
from fedjax.legacy.core.dataset_util import create_tf_dataset_for_clients
from fedjax.legacy.core.dataset_util import DatasetOrIterable
from fedjax.legacy.core.dataset_util import preprocess_tf_dataset
# Evaluation utilities
from fedjax.legacy.core.evaluation_util import aggregate_metrics
from fedjax.legacy.core.evaluation_util import evaluate_multiple_clients
from fedjax.legacy.core.evaluation_util import evaluate_single_client
# Federated algorithm.
from fedjax.legacy.core.federated_algorithm import FederatedAlgorithm
# Metrics.
from fedjax.legacy.core.metrics import accuracy_fn
from fedjax.legacy.core.metrics import CountMetric
from fedjax.legacy.core.metrics import cross_entropy_loss_fn
from fedjax.legacy.core.metrics import masked_accuracy_fn
from fedjax.legacy.core.metrics import masked_accuracy_fn_with_logits_mask
from fedjax.legacy.core.metrics import masked_count
from fedjax.legacy.core.metrics import masked_cross_entropy_loss_fn
from fedjax.legacy.core.metrics import MeanMetric
from fedjax.legacy.core.metrics import Metric
from fedjax.legacy.core.metrics import oov_rate
from fedjax.legacy.core.metrics import sequence_length
from fedjax.legacy.core.metrics import truncation_rate
# JAX model.
from fedjax.legacy.core.model import create_model_from_haiku
from fedjax.legacy.core.model import create_model_from_stax
from fedjax.legacy.core.model import Model
# JAX optimizer.
from fedjax.legacy.core.optimizer import get_optimizer
from fedjax.legacy.core.optimizer import Optimizer
from fedjax.legacy.core.optimizer import OptimizerName
# Dataset prefetching.
from fedjax.legacy.core.prefetch import PrefetchClientDatasetsIterator
# Regularization
from fedjax.legacy.core.regularizers import L2Regularizer
from fedjax.legacy.core.regularizers import Regularizer
# Serialization.
from fedjax.legacy.core.serialization import load_state
from fedjax.legacy.core.serialization import save_state
# PyTree utilities.
from fedjax.legacy.core.tree_util import tree_map
from fedjax.legacy.core.tree_util import tree_mean
from fedjax.legacy.core.tree_util import tree_multimap
# Common type definitions.
from fedjax.legacy.core.typing import Batch
from fedjax.legacy.core.typing import FederatedData
from fedjax.legacy.core.typing import MetricResults
from fedjax.legacy.core.typing import OptState
from fedjax.legacy.core.typing import Params
from fedjax.legacy.core.typing import PRNGKey
from fedjax.legacy.core.typing import PRNGSequence
from fedjax.legacy.core.typing import Updates
