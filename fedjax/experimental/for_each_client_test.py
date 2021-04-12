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
import numpy as np

# Map over clients and count how many points are greater than `limit` for
# each client. In addition to the total `count`, we'll also keep track of the
# `num` per step in our step results.


def client_init(server_state):
  return {'limit': server_state['limit'], 'count': 0.}


def client_step(client_state, batch):
  limit = client_state['limit']
  num = jnp.sum(batch['x'] > limit)
  client_state = {'limit': limit, 'count': client_state['count'] + num}
  step_result = {'num': num}
  return client_state, step_result


def client_final(client_state):
  return client_state['count']


# Variant of the task above where we provide a starting count for each client
# as part of persistent_client_state.


def client_init_with_persistent_state(server_state, persistent_client_state):
  return {
      'limit': server_state['limit'],
      'count': persistent_client_state['count'],
  }


class ForEachClientTest(absltest.TestCase):

  def test_for_each_client_jit(self):
    # Three clients with different data (`client_datasets`)
    # and starting counts (`client_infos`).
    client_datasets = [
        ('cid0', [{
            'x': jnp.array([1, 2, 3, 4])
        }, {
            'x': jnp.array([1, 2, 3])
        }]),
        ('cid1', [{
            'x': jnp.array([1, 2])
        }, {
            'x': jnp.array([1, 2, 3, 4, 5])
        }]),
        ('cid2', [{
            'x': jnp.array([1])
        }]),
    ]
    server_state = {'limit': jnp.array(2)}

    with self.subTest('without_persistent_client_state'):
      func = for_each_client.for_each_client_jit(client_init, client_step,
                                                 client_final)
      results = list(func(client_datasets, server_state))
      np.testing.assert_equal(results, [
          ('cid0', jnp.array(3), [{
              'num': jnp.array(2)
          }, {
              'num': jnp.array(1)
          }]),
          ('cid1', jnp.array(3), [{
              'num': jnp.array(0)
          }, {
              'num': jnp.array(3)
          }]),
          ('cid2', jnp.array(0), [{
              'num': jnp.array(0)
          }]),
      ])

    with self.subTest('with_persistent_client_state'):
      persistent_client_states = {
          'cid0': {
              'count': jnp.array(2)
          },
          'cid1': {
              'count': jnp.array(0)
          },
          'cid2': {
              'count': jnp.array(1)
          },
      }
      func = for_each_client.for_each_client_jit(
          client_init_with_persistent_state, client_step, client_final)
      results = list(
          func(client_datasets, server_state, persistent_client_states))
      np.testing.assert_equal(results, [
          ('cid0', jnp.array(5), [{
              'num': jnp.array(2)
          }, {
              'num': jnp.array(1)
          }]),
          ('cid1', jnp.array(3), [{
              'num': jnp.array(0)
          }, {
              'num': jnp.array(3)
          }]),
          ('cid2', jnp.array(1), [{
              'num': jnp.array(0)
          }]),
      ])


if __name__ == '__main__':
  absltest.main()
