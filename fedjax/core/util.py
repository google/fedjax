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
"""Small util functions with no better place to go."""

import jax.numpy as jnp


# Background: https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where
def safe_div(a, b):
  """Divides a by b, or returns 0 if b is 0 in a safe manner for gradient computation."""
  safe = b != 0
  c = a / jnp.where(safe, b, 1)
  return jnp.where(safe, c, 0)
