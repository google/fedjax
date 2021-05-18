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
"""Methods for serializing and deserializing arbitrary state pytrees."""

import pickle
from typing import Any

from absl import logging

import tensorflow as tf


def save_state(state: Any, path: str):
  """Saves state to file path."""
  logging.info('Saving state to %s.', path)
  with tf.io.gfile.GFile(path, 'wb') as f:
    pickle.dump(state, f)


def load_state(path: str) -> Any:
  """Loads saved state from file path."""
  logging.info('Loading params from %s.', path)
  with tf.io.gfile.GFile(path, 'rb') as f:
    return pickle.load(f)
