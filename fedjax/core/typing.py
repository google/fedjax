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
"""FedJAX types."""

from typing import Any, Dict, Mapping, Union
import haiku as hk
from jax.experimental import optimizers
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_federated as tff

# Flexible Params definition to support haiku as well as stax.
Params = Any
Updates = Params
# Flexible OptState definition to support optax and jax.experimental.optimizers.
OptState = Union[optax.OptState, optimizers.OptimizerState]

# Mapping of feature names to feature values.
Batch = Mapping[str, np.ndarray]
# Mapping of metric names to metric results.
MetricResults = Dict[str, jnp.ndarray]

# Missing JAX types.
PRNGKey = jnp.ndarray
# TODO(b/162863493): Replace with stricter definition if possible.
PyTree = Any

# Convenient FedJAX typecasts.
FederatedData = tff.simulation.ClientData
PRNGSequence = hk.PRNGSequence
