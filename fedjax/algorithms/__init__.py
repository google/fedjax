# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Federated learning algorithm implementations."""

from fedjax.algorithms.agnostic_fed_avg import AgnosticFedAvg
from fedjax.algorithms.agnostic_fed_avg import AgnosticFedAvgHParams
from fedjax.algorithms.agnostic_fed_avg import AgnosticFedAvgState
from fedjax.algorithms.agnostic_fed_avg import DomainAlgorithm
from fedjax.algorithms.fed_avg import FedAvg
from fedjax.algorithms.fed_avg import FedAvgHParams
from fedjax.algorithms.fed_avg import FedAvgState
from fedjax.algorithms.hyp_cluster import evaluate_multiple_clients_with_clusters
from fedjax.algorithms.hyp_cluster import HypCluster
from fedjax.algorithms.hyp_cluster import HypClusterHParams
from fedjax.algorithms.hyp_cluster import HypClusterState
from fedjax.algorithms.mime import Mime
from fedjax.algorithms.mime_lite import MimeLite
from fedjax.algorithms.mime_lite import MimeLiteClientTrainer
from fedjax.algorithms.mime_lite import MimeLiteHParams
from fedjax.algorithms.mime_lite import MimeLiteState
