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
"""Methods for checkpointing."""

import os.path
import re
from typing import Any, List, Optional, Tuple

from fedjax import core
import tensorflow as tf

_CHECKPOINT_PREFIX = 'checkpoint_'


def _get_checkpoint_paths(base_path: str) -> List[str]:
  """Returns all checkpoint paths present."""
  pattern = base_path + r'[0-9]{8}$'
  checkpoint_paths = []
  for path in tf.io.gfile.glob(base_path + '*'):
    if re.match(pattern, path):
      checkpoint_paths.append(path)

  def sort_key(path):
    return int(path.split(base_path)[-1])

  return sorted(checkpoint_paths, key=sort_key)


def load_latest_checkpoint(root_dir: str) -> Optional[Tuple[Any, int]]:
  """Loads latest checkpoint and round number."""
  base_path = os.path.join(root_dir, _CHECKPOINT_PREFIX)
  all_checkpoint_paths = _get_checkpoint_paths(base_path)
  if all_checkpoint_paths:
    latest_checkpoint_path = all_checkpoint_paths[-1]
    latest_round_num = int(latest_checkpoint_path.split(base_path)[-1])
    latest_state = core.load_state(latest_checkpoint_path)
    return latest_state, latest_round_num


def save_checkpoint(root_dir: str,
                    state: Any,
                    round_num: int = 0,
                    keep: int = 1):
  """Saves checkpoint and cleans up old checkpoints."""
  base_path = os.path.join(root_dir, _CHECKPOINT_PREFIX)
  checkpoint_path = f'{base_path}{round_num:08d}'
  core.save_state(state, checkpoint_path)
  remove_checkpoint_paths = _get_checkpoint_paths(base_path)[:-keep]
  for path in remove_checkpoint_paths:
    tf.io.gfile.remove(path)
