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
"""FedJAX legacy API."""

from fedjax.legacy import algorithms
from fedjax.legacy import datasets
from fedjax.legacy import models
from fedjax.legacy import training
from fedjax.legacy.core import *

# Please do not use symbols in `core` as they are not part of the public API.
try:
  del core  # pylint: disable=undefined-variable
except NameError:
  pass
