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
#
# Derived from https://github.com/google/flax/blob/master/flax/serialization.py
#
# Copyright 2021 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Simple serialization scheme based on msgpack.

We support serializing a dict and list of ndarrays holding numeric types, and
bytes (note tuples cannot be serialized since msgpack does not distinguish
tuples from lists during deserialization).
"""

import enum
import pickle

from absl import logging
from fedjax.core import util
import jax
import msgpack
import numpy as np

tf = util.import_tf()


def save_state(state, path):
  """Saves state to file path."""
  logging.info('Saving state to %s.', path)
  with tf.io.gfile.GFile(path, 'wb') as f:
    pickle.dump(state, f)


def load_state(path):
  """Loads saved state from file path."""
  logging.info('Loading params from %s.', path)
  with tf.io.gfile.GFile(path, 'rb') as f:
    return pickle.load(f)


# On-the-wire / disk serialization format

# We encode state-dicts via msgpack, using its custom type extension.
# https://github.com/msgpack/msgpack/blob/master/spec.md
#
# - ndarrays and DeviceArrays of numeric types are serialized to nested
#   msgpack-encoded string of (shape-tuple, dtype-name (e.g. 'float32'),
#   row-major array-bytes).
#
# - native complex scalars are converted to nested msgpack-encoded tuples
#   (real, imag).
#
# - ndarrays of bytes are serialized to nested msgpack-encoded string of
#   (shape-tuple, concatenated bytes).


def _ndarray_to_bytes(arr):
  """Save ndarray to simple msgpack encoding."""
  if isinstance(arr, jax.xla.DeviceArray):
    arr = np.array(arr)
  if arr.dtype.hasobject or arr.dtype.isalignedstruct:
    raise ValueError('Object and structured dtypes not supported '
                     'for serialization of ndarrays.')
  tpl = (arr.shape, arr.dtype.name, arr.tobytes('C'))
  return msgpack.packb(tpl, use_bin_type=True)


def _dtype_from_name(name):
  """Handle JAX bfloat16 dtype correctly."""
  if name == b'bfloat16':
    return jax.numpy.bfloat16
  else:
    return np.dtype(name)


def _ndarray_from_bytes(data):
  """Load ndarray from simple msgpack encoding."""
  shape, dtype_name, buffer = msgpack.unpackb(data, raw=True)
  return np.frombuffer(
      buffer, dtype=_dtype_from_name(dtype_name), count=-1, offset=0).reshape(
          shape, order='C')


def _bytes_ndarray_to_bytes(x):
  shape = x.shape
  flat = list(x.flatten())
  if flat and not isinstance(flat[0], bytes):
    raise ValueError('Only ndarrays holding bytes objects can be serialized.')
  tpl = shape, flat
  return msgpack.packb(tpl, use_bin_type=True)


def _object_ndarray_from_bytes(data):
  shape, flat = msgpack.unpackb(data, raw=True)
  return np.array(flat, dtype=object).reshape(shape)


class _MsgpackExtType(enum.IntEnum):
  """Messagepack custom type ids."""
  ndarray = 1
  native_complex = 2
  npscalar = 3
  bytes_ndarray = 4


def _msgpack_ext_pack(x):
  """Messagepack encoders for custom types."""
  if isinstance(x, np.ndarray) and x.dtype.hasobject:
    return msgpack.ExtType(_MsgpackExtType.bytes_ndarray,
                           _bytes_ndarray_to_bytes(x))
  elif isinstance(x, (np.ndarray, jax.xla.DeviceArray)):
    return msgpack.ExtType(_MsgpackExtType.ndarray, _ndarray_to_bytes(x))
  elif np.issctype(type(x)):
    # pack scalar as ndarray
    return msgpack.ExtType(_MsgpackExtType.npscalar,
                           _ndarray_to_bytes(np.asarray(x)))
  elif isinstance(x, complex):
    return msgpack.ExtType(_MsgpackExtType.native_complex,
                           msgpack.packb((x.real, x.imag)))
  print('falling back', repr(x))
  return x


def _msgpack_ext_unpack(code, data):
  """Messagepack decoders for custom types."""
  if code == _MsgpackExtType.ndarray:
    return _ndarray_from_bytes(data)
  elif code == _MsgpackExtType.native_complex:
    complex_tuple = msgpack.unpackb(data)
    return complex(complex_tuple[0], complex_tuple[1])
  elif code == _MsgpackExtType.npscalar:
    ar = _ndarray_from_bytes(data)
    return ar[()]  # unpack ndarray to scalar
  elif code == _MsgpackExtType.bytes_ndarray:
    return _object_ndarray_from_bytes(data)
  return msgpack.ExtType(code, data)


# User-facing API calls:


def msgpack_serialize(pytree):
  """Save data structure to bytes in msgpack format.

  Low-level function that only supports python trees with array leaves,
  for custom objects use `to_bytes`.

  Args:
    pytree: python tree of dict, list with python primitives and array leaves.

  Returns:
    msgpack-encoded bytes of pytree.
  """
  return msgpack.packb(pytree, default=_msgpack_ext_pack, strict_types=True)


def msgpack_deserialize(encoded_pytree):
  """Restore data structure from bytes in msgpack format.

  Low-level function that only supports python trees with array leaves,
  for custom objects use `from_bytes`.

  Args:
    encoded_pytree: msgpack-encoded bytes of python tree.

  Returns:
    Python tree of dict, list with python primitive and array leaves.
  """
  return msgpack.unpackb(
      encoded_pytree, ext_hook=_msgpack_ext_unpack, raw=False)
