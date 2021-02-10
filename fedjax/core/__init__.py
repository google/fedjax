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
from fedjax.core import test_util

# Client trainer.
from fedjax.core.client_trainer import ClientTrainer
from fedjax.core.client_trainer import ControlVariateTrainer
from fedjax.core.client_trainer import ControlVariateTrainerState
from fedjax.core.client_trainer import DefaultClientTrainer
from fedjax.core.client_trainer import DefaultClientTrainerState
from fedjax.core.client_trainer import train_multiple_clients
from fedjax.core.client_trainer import train_single_client
# Dataset utilities.
from fedjax.core.dataset_util import ClientDataHParams
from fedjax.core.dataset_util import create_tf_dataset_for_clients
from fedjax.core.dataset_util import DatasetOrIterable
from fedjax.core.dataset_util import preprocess_tf_dataset
# Evaluation utilities
from fedjax.core.evaluation_util import aggregate_metrics
from fedjax.core.evaluation_util import evaluate_multiple_clients
from fedjax.core.evaluation_util import evaluate_single_client
# Federated algorithm.
from fedjax.core.federated_algorithm import FederatedAlgorithm
# Metrics.
from fedjax.core.metrics import accuracy_fn
from fedjax.core.metrics import CountMetric
from fedjax.core.metrics import cross_entropy_loss_fn
from fedjax.core.metrics import masked_accuracy_fn
from fedjax.core.metrics import masked_accuracy_fn_with_logits_mask
from fedjax.core.metrics import masked_count
from fedjax.core.metrics import masked_cross_entropy_loss_fn
from fedjax.core.metrics import MeanMetric
from fedjax.core.metrics import Metric
from fedjax.core.metrics import oov_rate
from fedjax.core.metrics import sequence_length
from fedjax.core.metrics import truncation_rate
# JAX model.
from fedjax.core.model import create_model_from_haiku
from fedjax.core.model import create_model_from_stax
from fedjax.core.model import Model
# JAX optimizer.
from fedjax.core.optimizer import get_optimizer
from fedjax.core.optimizer import Optimizer
from fedjax.core.optimizer import OptimizerName
# Dataset prefetching.
from fedjax.core.prefetch import PrefetchClientDatasetsIterator
# Regularization
from fedjax.core.regularizers import L2Regularizer
from fedjax.core.regularizers import Regularizer
# Serialization.
from fedjax.core.serialization import load_state
from fedjax.core.serialization import save_state
# PyTree utilities.
from fedjax.core.tree_util import tree_map
from fedjax.core.tree_util import tree_mean
from fedjax.core.tree_util import tree_multimap
# Common type definitions.
from fedjax.core.typing import Batch
from fedjax.core.typing import FederatedData
from fedjax.core.typing import MetricResults
from fedjax.core.typing import OptState
from fedjax.core.typing import Params
from fedjax.core.typing import PRNGKey
from fedjax.core.typing import PRNGSequence
from fedjax.core.typing import Updates
