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
"""Tests for fedjax.experimental.for_each_client."""

from absl.testing import absltest

from fedjax.experimental import for_each_client

import jax
import jax.numpy as jnp
import numpy.testing as npt

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

  def test_basic_output(self):
    func = for_each_client.for_each_client_jit(client_init,
                                               client_step_with_result,
                                               client_final)
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

    func = for_each_client.for_each_client_jit(my_client_init, my_client_step,
                                               my_client_final)
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

  def test_basic_output(self):
    func = for_each_client.for_each_client_debug(client_init,
                                                 client_step_with_result,
                                                 client_final)
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

    func = for_each_client.for_each_client_debug(my_client_init, my_client_step,
                                                 my_client_final)
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

    func = for_each_client.for_each_client_debug(my_client_init,
                                                 client_step_with_result,
                                                 client_final)
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

    func = for_each_client.for_each_client_debug(client_init, my_client_step,
                                                 client_final)
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

    func = for_each_client.for_each_client_debug(client_init,
                                                 client_step_with_result,
                                                 my_client_final)
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


if __name__ == '__main__':
  absltest.main()
