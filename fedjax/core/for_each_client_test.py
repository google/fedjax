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
"""Tests for fedjax.core.for_each_client."""

import os

from absl.testing import absltest

from fedjax.core import for_each_client

import jax
from jax.lib import xla_bridge
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt


def setUpModule():
  """Run all tests with 8 CPU devices."""
  global prev_xla_flags  # pylint: disable=global-variable-undefined
  prev_xla_flags = os.getenv('XLA_FLAGS')
  flags_str = prev_xla_flags or ''
  # Don't override user-specified device count, or other XLA flags.
  if 'xla_force_host_platform_device_count' not in flags_str:
    os.environ['XLA_FLAGS'] = (
        flags_str + ' --xla_force_host_platform_device_count=8')
  # Clear any cached backends so new CPU backend will pick up the env var.
  xla_bridge.get_backend.cache_clear()


def tearDownModule():
  """Reset to previous configuration in case other test modules will be run."""
  if prev_xla_flags is None:
    del os.environ['XLA_FLAGS']
  else:
    os.environ['XLA_FLAGS'] = prev_xla_flags
  xla_bridge.get_backend.cache_clear()


# Map over clients and count how many points are greater than `limit` for
# each client. Each client also has a different `start` that is specified via
# client input.


def client_init(shared_input, client_input):
  client_step_state = {
      'limit': shared_input['limit'],
      'count': client_input['start']
  }
  return client_step_state


def client_step(client_step_state, batch):
  num = jnp.sum(batch['x'] > client_step_state['limit'])
  client_step_state = {
      'limit': client_step_state['limit'],
      'count': client_step_state['count'] + num
  }
  return client_step_state


def client_final(shared_input, client_step_state):
  del shared_input  # Unused.
  return client_step_state['count']


# We'll also keep track of the `num` per step in our step results.


def client_step_with_result(client_step_state, batch):
  num = jnp.sum(batch['x'] > client_step_state['limit'])
  client_step_state = {
      'limit': client_step_state['limit'],
      'count': client_step_state['count'] + num
  }
  client_step_result = {'num': num}
  return client_step_state, client_step_result


class DoNotRun:

  class BaseTest(absltest.TestCase):

    @classmethod
    def setUpClass(cls):
      super().setUpClass()
      cls.SHARED_INPUT = {'limit': jnp.array(2)}
      # Three clients with different data and different starting counts.
      cls.CLIENTS = [
          (b'cid0', [{
              'x': jnp.array([1, 2, 3, 4])
          }, {
              'x': jnp.array([1, 2, 3])
          }], {
              'start': jnp.array(2)
          }),
          (b'cid1', [{
              'x': jnp.array([1, 2])
          }, {
              'x': jnp.array([1, 2, 3, 4, 5])
          }], {
              'start': jnp.array(0)
          }),
          (b'cid2', [{
              'x': jnp.array([1])
          }], {
              'start': jnp.array(1)
          }),
      ]
      cls.EXPECTED_WITHOUT_STEP_RESULT = [(b'cid0', jnp.array(5)),
                                          (b'cid1', jnp.array(3)),
                                          (b'cid2', jnp.array(1))]
      cls.EXPECTED_WITH_STEP_RESULT = [
          (b'cid0', jnp.array(5), [{
              'num': jnp.array(2)
          }, {
              'num': jnp.array(1)
          }]),
          (b'cid1', jnp.array(3), [{
              'num': jnp.array(0)
          }, {
              'num': jnp.array(3)
          }]),
          (b'cid2', jnp.array(1), [{
              'num': jnp.array(0)
          }]),
      ]


class ForEachClientTest(DoNotRun.BaseTest):

  def test_without_step_result(self):
    func = for_each_client.for_each_client(client_init, client_step,
                                           client_final)
    results = list(func(self.SHARED_INPUT, self.CLIENTS))
    npt.assert_equal(results, self.EXPECTED_WITHOUT_STEP_RESULT)

  def test_with_step_result(self):
    func = for_each_client.for_each_client(
        client_init,
        client_step_with_result,
        client_final,
        with_step_result=True)
    results = list(func(self.SHARED_INPUT, self.CLIENTS))
    npt.assert_equal(results, self.EXPECTED_WITH_STEP_RESULT)


class ForEachClientJitTest(DoNotRun.BaseTest):

  def setUp(self):
    super().setUp()
    self._backend = for_each_client.ForEachClientJitBackend()

  def test_basic_output(self):
    func = self._backend(client_init, client_step_with_result, client_final)
    results = list(func(self.SHARED_INPUT, self.CLIENTS))
    npt.assert_equal(results, self.EXPECTED_WITH_STEP_RESULT)

  def test_has_jit(self):
    num_calls = [0, 0, 0]

    def my_client_init(*args, **kwargs):
      num_calls[0] += 1
      return client_init(*args, **kwargs)

    def my_client_step(*args, **kwargs):
      num_calls[1] += 1
      return client_step_with_result(*args, **kwargs)

    def my_client_final(*args, **kwargs):
      num_calls[2] += 1
      return client_final(*args, **kwargs)

    func = self._backend(my_client_init, my_client_step, my_client_final)
    npt.assert_equal(
        list(func(self.SHARED_INPUT, self.CLIENTS)),
        self.EXPECTED_WITH_STEP_RESULT)
    self.assertListEqual(num_calls, [1, 5, 1])
    # Has jit, so repeated calls will not increase num_calls.
    npt.assert_equal(
        list(func(self.SHARED_INPUT, self.CLIENTS)),
        self.EXPECTED_WITH_STEP_RESULT)
    self.assertListEqual(num_calls, [1, 5, 1])


class ForEachClientDebugTest(DoNotRun.BaseTest):

  def setUp(self):
    super().setUp()
    self._backend = for_each_client.ForEachClientDebugBackend()

  def test_basic_output(self):
    func = self._backend(client_init, client_step_with_result, client_final)
    results = list(func(self.SHARED_INPUT, self.CLIENTS))
    npt.assert_equal(results, self.EXPECTED_WITH_STEP_RESULT)

  def test_no_jit(self):
    num_calls = [0, 0, 0]

    def my_client_init(*args, **kwargs):
      num_calls[0] += 1
      return client_init(*args, **kwargs)

    def my_client_step(*args, **kwargs):
      num_calls[1] += 1
      return client_step_with_result(*args, **kwargs)

    def my_client_final(*args, **kwargs):
      num_calls[2] += 1
      return client_final(*args, **kwargs)

    func = self._backend(my_client_init, my_client_step, my_client_final)
    npt.assert_equal(
        list(func(self.SHARED_INPUT, self.CLIENTS)),
        self.EXPECTED_WITH_STEP_RESULT)
    self.assertListEqual(num_calls, [3, 5, 3])
    # No jit, so repeated calls will increase num_calls.
    npt.assert_equal(
        list(func(self.SHARED_INPUT, self.CLIENTS)),
        self.EXPECTED_WITH_STEP_RESULT)
    self.assertListEqual(num_calls, [6, 10, 6])

  def test_client_init_error(self):

    def my_client_init(shared_input, client_input):
      if client_input['start'].copy() == 0:
        raise ValueError('Oops')
      return client_init(shared_input, client_input)

    func = self._backend(my_client_init, client_step_with_result, client_final)
    with self.assertRaisesRegex(
        for_each_client.ForEachClientError,
        'Stage: client_init.*Base error is ValueError: Oops') as cm:
      list(func(self.SHARED_INPUT, self.CLIENTS))
    # At least one side of the comparison of npt.assert_equal needs to be
    # np.ndarray to trigger npt.assert_array_equal, thus the device_get calls.
    npt.assert_equal(
        cm.exception.context, {
            'client_id': b'cid1',
            'client_init': my_client_init,
            'shared_input': jax.device_get(self.SHARED_INPUT),
            'client_input': jax.device_get({'start': jnp.array(0)})
        })

  def test_client_step_error(self):

    def my_client_step(state, batch):
      if len(batch['x']) == 3:
        raise ValueError('Oops')
      return client_step_with_result(state, batch)

    func = self._backend(client_init, my_client_step, client_final)
    with self.assertRaisesRegex(
        for_each_client.ForEachClientError,
        r'Stage: client_step.*Base error is ValueError: Oops') as cm:
      list(func(self.SHARED_INPUT, self.CLIENTS))
    # At least one side of the comparison of npt.assert_equal needs to be
    # np.ndarray to trigger npt.assert_array_equal, thus the device_get calls.
    npt.assert_equal(
        cm.exception.context, {
            'client_id':
                b'cid0',
            'client_step':
                my_client_step,
            'state':
                jax.device_get({
                    'limit': jnp.array(2),
                    'count': jnp.array(4)
                }),
            'batch':
                jax.device_get({'x': jnp.array([1, 2, 3])})
        })

  def test_client_final_error(self):

    def my_client_final(shared_input, state):
      if state['count'].copy() == 1:
        raise ValueError('Oops')
      return client_final(shared_input, state)

    func = self._backend(client_init, client_step_with_result, my_client_final)
    with self.assertRaisesRegex(
        for_each_client.ForEachClientError,
        r'Stage: client_final.*Base error is ValueError: Oops') as cm:
      list(func(self.SHARED_INPUT, self.CLIENTS))
    # At least one side of the comparison of npt.assert_equal needs to be
    # np.ndarray to trigger npt.assert_array_equal, thus the device_get calls.
    npt.assert_equal(
        cm.exception.context, {
            'client_id':
                b'cid2',
            'client_final':
                my_client_final,
            'shared_input':
                jax.device_get(self.SHARED_INPUT),
            'state':
                jax.device_get({
                    'limit': jnp.array(2),
                    'count': jnp.array(1)
                })
        })


class BlockifyTest(absltest.TestCase):

  def test_blockify(self):
    clients = [
        ('a', np.random.uniform(size=(1, 8)), np.random.uniform(size=(2,))),
        ('b', np.random.uniform(size=(4, 8)), np.random.uniform(size=(2,))),
        ('c', np.random.uniform(size=(3, 8)), np.random.uniform(size=(2,))),
        ('d', np.random.uniform(size=(2, 8)), np.random.uniform(size=(2,)))
    ]
    a, b, c, d = [clients[i][1] for i in range(4)]

    with self.subTest('no padding client'):
      blocks = list(for_each_client._blockify(clients, 2))
      self.assertLen(blocks, 2)
      # block 0
      self.assertListEqual(blocks[0].client_id, ['b', 'c'])
      self.assertListEqual(blocks[0].client_mask, [True, True])
      self.assertListEqual(blocks[0].num_batches, [4, 3])
      npt.assert_equal(blocks[0].masked_batches,
                       [([b[0], c[0]], [True, True]),
                        ([b[1], c[1]], [True, True]),
                        ([b[2], c[2]], [True, True]),
                        ([b[3], np.zeros_like(b[0])], [True, False])])
      npt.assert_equal(blocks[0].client_input, [clients[1][-1], clients[2][-1]])
      # block 1
      self.assertListEqual(blocks[1].client_id, ['d', 'a'])
      self.assertListEqual(blocks[1].client_mask, [True, True])
      self.assertListEqual(blocks[1].num_batches, [2, 1])
      npt.assert_equal(blocks[1].masked_batches,
                       [([d[0], a[0]], [True, True]),
                        ([d[1], np.zeros_like(d[0])], [True, False])])
      npt.assert_equal(blocks[1].client_input, [clients[3][-1], clients[0][-1]])

    with self.subTest('has padding client'):
      blocks = list(for_each_client._blockify(clients, 3))
      self.assertLen(blocks, 2)
      # block 0
      self.assertListEqual(blocks[0].client_id, ['b', 'c', 'd'])
      self.assertListEqual(blocks[0].client_mask, [True, True, True])
      self.assertListEqual(blocks[0].num_batches, [4, 3, 2])
      npt.assert_equal(
          blocks[0].masked_batches,
          [([b[0], c[0], d[0]], [True, True, True]),
           ([b[1], c[1], d[1]], [True, True, True]),
           ([b[2], c[2], np.zeros_like(b[0])], [True, True, False]),
           ([b[3], np.zeros_like(b[0]),
             np.zeros_like(b[0])], [True, False, False])])
      npt.assert_equal(blocks[0].client_input,
                       [clients[1][-1], clients[2][-1], clients[3][-1]])
      # block 1
      self.assertListEqual(blocks[1].client_id, ['a', None, None])
      self.assertListEqual(blocks[1].client_mask, [True, False, False])
      self.assertListEqual(blocks[1].num_batches, [1, 0, 0])
      npt.assert_equal(blocks[1].masked_batches,
                       [([a[0], np.zeros_like(a[0]),
                          np.zeros_like(a[0])], [True, False, False])])
      npt.assert_equal(blocks[1].client_input, [
          clients[0][-1],
          np.zeros_like(clients[0][-1]),
          np.zeros_like(clients[0][-1])
      ])

  def test_blockify_zero_batches(self):
    blocks = list(for_each_client._blockify([('a', [], np.array(1))], 3))
    self.assertLen(blocks, 1)
    self.assertListEqual(blocks[0].client_id, ['a', None, None])
    self.assertListEqual(blocks[0].client_mask, [True, False, False])
    self.assertListEqual(blocks[0].num_batches, [0, 0, 0])
    self.assertListEqual(blocks[0].masked_batches, [])
    npt.assert_equal(
        blocks[0].client_input,
        [np.array(1), np.array(0), np.array(0)])


class ForEachClientPmapTest(absltest.TestCase):

  def test_for_each_client_pmap(self):
    # Make sure setUpModule() does the work.
    self.assertEqual(jax.local_device_count(), 8)

    def my_client_init(shared_input, client_input):
      return {'x': jnp.dot(shared_input['y'], client_input['z'])}

    def my_client_step(state, batch):
      return {'x': jnp.dot(state['x'], batch['w'])}, {'u': jnp.sum(batch['w'])}

    def my_client_final(shared_input, state):
      return {'v': jnp.dot(state['x'], shared_input['y'])}

    shared_input = {'y': np.random.uniform(size=[16, 16])}
    clients = []
    for i in range(10):
      client_id = i
      client_input = {'z': np.random.uniform(size=[16, 16])}
      client_batches = []
      for _ in range(i):
        client_batches.append({'w': np.random.uniform(size=[16, 16])})
      clients.append((client_id, client_batches, client_input))

    expected = {}
    for client_id, client_output, step_results in (
        for_each_client.ForEachClientJitBackend()(my_client_init,
                                                  my_client_step,
                                                  my_client_final)(shared_input,
                                                                   clients)):
      expected[client_id] = jax.device_get((client_output, step_results))

    for i in range(jax.local_device_count() + 1):
      with self.subTest(f'{i} devices' if i > 0 else 'default devices'):
        if i > 0:
          devices = jax.local_devices()[:i]
        else:
          devices = None
        actual = {}
        for client_id, client_output, step_results in (
            for_each_client.ForEachClientPmapBackend(devices)(
                my_client_init, my_client_step, my_client_final)(shared_input,
                                                                 clients)):
          actual[client_id] = (client_output, step_results)
        jax.tree_util.tree_multimap(npt.assert_allclose, actual, expected)
        # Check actual can be operated over.
        jax.tree_util.tree_multimap(
            npt.assert_allclose,
            *jax.tree_util.tree_map(lambda x: x + 1, (actual, expected)))


class BackendChoiceTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._backend = for_each_client.ForEachClientJitBackend()

  def tearDown(self):
    super().tearDown()
    for_each_client.set_for_each_client_backend(None)

  def test_default_backend(self):
    self.assertIsInstance(for_each_client.get_for_each_client_backend(),
                          for_each_client.ForEachClientJitBackend)

  def test_set_and_get_concrete(self):
    self.assertIsNot(for_each_client.get_for_each_client_backend(),
                     self._backend)
    for_each_client.set_for_each_client_backend(self._backend)
    self.assertIs(for_each_client.get_for_each_client_backend(), self._backend)

  def test_set_and_get_str(self):
    with self.subTest('debug'):
      for_each_client.set_for_each_client_backend('debug')
      self.assertIsInstance(for_each_client.get_for_each_client_backend(),
                            for_each_client.ForEachClientDebugBackend)
    with self.subTest('jit'):
      for_each_client.set_for_each_client_backend('jit')
      self.assertIsInstance(for_each_client.get_for_each_client_backend(),
                            for_each_client.ForEachClientJitBackend)
    with self.subTest('pmap'):
      for_each_client.set_for_each_client_backend('pmap')
      self.assertIsInstance(for_each_client.get_for_each_client_backend(),
                            for_each_client.ForEachClientPmapBackend)
    with self.subTest('invalid'):
      with self.assertRaisesRegex(ValueError, "Unsupported backend 'invalid'"):
        for_each_client.set_for_each_client_backend('invalid')

  def test_context_manager(self):
    with for_each_client.for_each_client_backend(self._backend):
      self.assertIs(for_each_client.get_for_each_client_backend(),
                    self._backend)
      with for_each_client.for_each_client_backend('debug'):
        self.assertIsInstance(for_each_client.get_for_each_client_backend(),
                              for_each_client.ForEachClientDebugBackend)
      self.assertIs(for_each_client.get_for_each_client_backend(),
                    self._backend)


if __name__ == '__main__':
  absltest.main()
