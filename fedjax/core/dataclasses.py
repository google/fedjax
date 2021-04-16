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
"""Slimmed down dataclass."""

import dataclasses
import jax


# Forked and slimmed down from
# https://flax.readthedocs.io/en/latest/_modules/flax/struct.html#dataclass
# https://github.com/google/jax/issues/2371
def dataclass(clz: type):
  """Creates a dataclass which can be passed to functional transformations."""
  data_clz = dataclasses.dataclass(frozen=True)(clz)
  meta_fields = []
  data_fields = []
  for name, field_info in data_clz.__dataclass_fields__.items():
    is_pytree_node = field_info.metadata.get('pytree_node', True)
    if is_pytree_node:
      data_fields.append(name)
    else:
      meta_fields.append(name)

  def replace(self, **updates):
    """"Returns a new object replacing the specified fields with new values."""
    return dataclasses.replace(self, **updates)

  data_clz.replace = replace

  def iterate_clz(x):
    meta = tuple(getattr(x, name) for name in meta_fields)
    data = tuple(getattr(x, name) for name in data_fields)
    return data, meta

  def clz_from_iterable(meta, data):
    meta_args = tuple(zip(meta_fields, meta))
    data_args = tuple(zip(data_fields, data))
    kwargs = dict(meta_args + data_args)
    return data_clz(**kwargs)

  jax.tree_util.register_pytree_node(data_clz, iterate_clz, clz_from_iterable)
  return data_clz
