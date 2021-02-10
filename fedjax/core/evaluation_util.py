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
"""Functions for evaluation."""

import collections
from typing import Dict, Iterable, Iterator, List, Optional

from fedjax.core import dataset_util
from fedjax.core import metrics
from fedjax.core import prefetch
from fedjax.core.model import Model
from fedjax.core.typing import FederatedData
from fedjax.core.typing import MetricResults
from fedjax.core.typing import Params


def aggregate_metrics(
    metrics_iter: Iterable[Dict[str, metrics.Metric]]) -> MetricResults:
  """Aggregates multiple metrics into a single overall metrics output.

  Args:
    metrics_iter: Multiple mappings of metrics.

  Returns:
    Aggregated result metrics or empty mapping if `metrics_iter` is empty.
  """
  metrics_iter = iter(metrics_iter)
  try:
    aggregated_metrics = next(metrics_iter)
  except StopIteration:
    return collections.OrderedDict()

  for metrics_dict in metrics_iter:
    next_aggregated_metrics = collections.OrderedDict()
    for name, metric in metrics_dict.items():
      next_aggregated_metrics[name] = aggregated_metrics[name].merge(metric)
    aggregated_metrics = next_aggregated_metrics

  metrics_results = collections.OrderedDict()
  for name, metric in aggregated_metrics.items():
    metrics_results[name] = metric.result()

  return metrics_results


def evaluate_single_client(dataset: dataset_util.DatasetOrIterable,
                           model: Model, params: Params) -> MetricResults:
  """Evaluates model for a single client's dataset.

  Args:
    dataset: Dataset of ordered mapping of examples [client_batch_size, d].
    model: Model implementation.
    params: Model parameters.

  Returns:
    Ordered mapping of client metrics or empty mapping if the dataset is empty.
  """

  def compute_batch_metrics(batch):
    return model.evaluate(params, batch)

  batch_metrics_iterator = map(compute_batch_metrics,
                               dataset_util.iterate(dataset))
  return aggregate_metrics(batch_metrics_iterator)


def evaluate_multiple_clients(
    federated_data: FederatedData,
    client_ids: List[str],
    model: Model,
    params: Params,
    client_data_hparams: Optional[dataset_util.ClientDataHParams] = None
) -> Iterator[MetricResults]:
  """Evaluates model over input clients' datasets.

  Args:
    federated_data: Federated data separated per client.
    client_ids: Ids of clients to evaluate.
    model: Model implementation.
    params: Pytree of model parameters to be evaluated.
    client_data_hparams: Hyperparameters for client dataset preparation.

  Yields:
    Ordered mapping of metrics per client or empty mapping if a client's dataset
      is emtpy.
  """
  if client_data_hparams is not None:
    federated_data = federated_data.preprocess(
        lambda ds: dataset_util.preprocess_tf_dataset(ds, client_data_hparams))
  for _, client_dataset in prefetch.PrefetchClientDatasetsIterator(
      federated_data, client_ids):
    yield evaluate_single_client(client_dataset, model, params)
