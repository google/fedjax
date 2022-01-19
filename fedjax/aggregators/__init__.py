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
"""FedJAX aggregators."""

# Default aggregators.
from fedjax.aggregators.aggregator import Aggregator
from fedjax.aggregators.aggregator import mean_aggregator

# Compression aggregators.
from fedjax.aggregators.compression import binary_stochastic_quantize
from fedjax.aggregators.compression import uniform_stochastic_quantize
from fedjax.aggregators.compression import uniform_stochastic_quantize_pytree
from fedjax.aggregators.compression import uniform_stochastic_quantizer

# Hadamard transform.
from fedjax.aggregators.walsh_hadamard import walsh_hadamard_transform
