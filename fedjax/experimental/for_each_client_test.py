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

import jax.numpy as jnp
import numpy.testing as npt

# Map over clients and count how many points are greater than `limit` for
# each client. In addition to the total `count`, we'll also keep track of the
# `num` per step in our step results. Each client also has a different `start`
# that is specified via persistent client state.


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
  client_step_result = {'num': num}
  return client_step_state, client_step_result


def client_final(shared_input, client_step_state):
  del shared_input  # Unused.
  return client_step_state['count']


class ForEachClientTest(absltest.TestCase):

  def test_for_each_client_jit(self):
    shared_input = {'limit': jnp.array(2)}
    # Three clients with different data and different starting counts.
    clients = [
        (b'cid0',
         [{'x': jnp.array([1, 2, 3, 4])}, {'x': jnp.array([1, 2, 3])}],
         {'start': jnp.array(2)}),
        (b'cid1',
         [{'x': jnp.array([1, 2])}, {'x': jnp.array([1, 2, 3, 4, 5])}],
         {'start': jnp.array(0)}),
        (b'cid2',
         [{'x': jnp.array([1])}],
         {'start': jnp.array(1)}),
    ]
    func = for_each_client.for_each_client_jit(client_init, client_step,
                                               client_final)
    results = list(func(clients, shared_input))
    npt.assert_equal(results, [
        (b'cid0', jnp.array(5),
         [{'num': jnp.array(2)}, {'num': jnp.array(1)}]),
        (b'cid1', jnp.array(3),
         [{'num': jnp.array(0)}, {'num': jnp.array(3)}]),
        (b'cid2', jnp.array(1),
         [{'num': jnp.array(0)}]),
    ])


if __name__ == '__main__':
  absltest.main()
