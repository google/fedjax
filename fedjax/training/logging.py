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
"""Logger component."""

import collections
import os.path
from typing import Any, Optional

from absl import logging

import tensorflow as tf


class Logger:
  """Class to encapsulate tf.summary.SummaryWriter logging logic."""

  def __init__(self, root_dir: Optional[str] = None):
    """Initializes summary writers and log directory."""
    self._root_dir = root_dir
    self._summary_writers = collections.OrderedDict()

  def log(self, writer_name: str, metric_name: str, metric_value: Any,
          round_num: int):
    """Records metric using specified summary writer.

    Logs at INFO verbosity. If root_dir is set and metric_value is a numeric
    scalar value, convertible to a float32 Tensor, also writes scalar summary.

    Args:
      writer_name: Name of summary writer.
      metric_name: Name of metric to log and optionally write scalar summary.
      metric_value: Value of metric to log and optionally write scalar summary.
      round_num: Round number to log and optionally write scalar summary.
    """
    logging.info('round %d %s: %s = %s', round_num, writer_name, metric_name,
                 metric_value)

    if self._root_dir is None:
      return

    if writer_name not in self._summary_writers:
      self._summary_writers[writer_name] = tf.summary.create_file_writer(
          os.path.join(self._root_dir, writer_name))

    with self._summary_writers[writer_name].as_default():
      try:
        tf.summary.scalar(metric_name, metric_value, step=round_num)
      except (ValueError, tf.errors.UnimplementedError) as e:
        logging.info('Failed to log summary with exception %s', e)
        pass
