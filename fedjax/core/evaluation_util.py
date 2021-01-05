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
from typing import Iterable, Iterator, List, Optional

from fedjax.core import dataset_util
from fedjax.core import tree_util
from fedjax.core.model import Model
from fedjax.core.typing import FederatedData
from fedjax.core.typing import Metrics
from fedjax.core.typing import Params

_WEIGHT = 'weight'


def aggregate_metrics(metrics: Iterable[Metrics]) -> Metrics:
  """Aggregates multiple metrics into a single overall metrics output.

  Args:
    metrics: Multiple mappings of metric names to values with the weight metric
      given by 'weight'.

  Returns:
    Summarized metrics of identical structure to `metrics` or empty mapping if
      `metrics` is empty.
  """
  metrics_iter = iter(metrics)
  try:
    m = next(metrics_iter)
    weighted_metrics = tree_util.tree_weight(m, m[_WEIGHT])
    weight = m[_WEIGHT]
  except StopIteration:
    return collections.OrderedDict()

  for m in metrics_iter:
    weighted_metrics = tree_util.tree_sum(weighted_metrics,
                                          tree_util.tree_weight(m, m[_WEIGHT]))
    weight += m[_WEIGHT]
  inverse_weight = 1. / weight if weight > 0 else 0.
  metrics = tree_util.tree_weight(weighted_metrics, inverse_weight)
  # TODO(jaero): Separate definition between counters (like weight) and metrics.
  # Since counters should just be summed w/o weighting.
  metrics[_WEIGHT] = weight
  return metrics


def evaluate_single_client(dataset: dataset_util.DatasetOrIterable,
                           model: Model, params: Params) -> Metrics:
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


# TODO(jaero): Mark data_hparams as required if we plan to pmap this.
def evaluate_multiple_clients(
    federated_data: FederatedData,
    client_ids: List[str],
    model: Model,
    params: Params,
    client_data_hparams: Optional[dataset_util.ClientDataHParams] = None
) -> Iterator[Metrics]:
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
  for client_id in client_ids:
    client_dataset = federated_data.create_tf_dataset_for_client(client_id)
    if client_data_hparams is not None:
      client_dataset = dataset_util.preprocess_tf_dataset(
          client_dataset, client_data_hparams)
    yield evaluate_single_client(client_dataset, model, params)
