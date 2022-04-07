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
"""Registry of standard tasks.

Each task is represented as a (train federated data, test federated data, model)
tuple.
"""

from typing import Optional, Tuple

from fedjax import datasets
from fedjax import models
from fedjax.core import federated_data
from fedjax.core import models as core_models

ALL_TASKS = ('EMNIST_CONV', 'EMNIST_LOGISTIC', 'EMNIST_DENSE',
             'SHAKESPEARE_CHARACTER', 'STACKOVERFLOW_WORD', 'CIFAR100_LOGISTIC')


def get_task(
    name: str,
    mode: str = 'sqlite',
    cache_dir: Optional[str] = None
) -> Tuple[federated_data.FederatedData, federated_data.FederatedData,
           core_models.Model]:
  """Gets a standard task.

  Args:
    name: Name of the task to get. Must be one of fedjax.training.ALL_TASKS.
    mode: 'sqlite'.
    cache_dir: Directory to cache files in 'sqlite' mode.

  Returns:
    (train federated data, test federated data, model) tuple.
  """
  if name == 'EMNIST_CONV':
    train, test = datasets.emnist.load_data(
        only_digits=False, mode=mode, cache_dir=cache_dir)
    model = models.emnist.create_conv_model(only_digits=False)
  elif name == 'EMNIST_LOGISTIC':
    train, test = datasets.emnist.load_data(
        only_digits=False, mode=mode, cache_dir=cache_dir)
    model = models.emnist.create_logistic_model(only_digits=False)
  elif name == 'EMNIST_DENSE':
    train, test = datasets.emnist.load_data(
        only_digits=False, mode=mode, cache_dir=cache_dir)
    model = models.emnist.create_dense_model(only_digits=False)
  elif name == 'SHAKESPEARE_CHARACTER':
    train, test = datasets.shakespeare.load_data(mode=mode, cache_dir=cache_dir)
    model = models.shakespeare.create_lstm_model()
  elif name == 'STACKOVERFLOW_WORD':
    train, _, test = datasets.stackoverflow.load_data(
        mode=mode, cache_dir=cache_dir)
    tokenizer = datasets.stackoverflow.StackoverflowTokenizer()
    # Same max sequence length as https://arxiv.org/pdf/2003.00295.pdf.
    max_length = 20
    train = train.preprocess_batch(tokenizer.as_preprocess_batch(max_length))
    test = test.preprocess_batch(tokenizer.as_preprocess_batch(max_length))
    model = models.stackoverflow.create_lstm_model(expected_length=13.3)
  elif name == 'CIFAR100_LOGISTIC':
    train, test = datasets.cifar100.load_data(mode=mode, cache_dir=cache_dir)
    train = train.preprocess_batch(datasets.cifar100.preprocess_batch_tff)
    test = test.preprocess_batch(datasets.cifar100.preprocess_batch_tff)
    model = models.cifar100.create_logistic_model()
  else:
    raise ValueError(f'Unsupported task: {name!r}')
  return train, test, model
