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
"""Tests for fedjax.core.client_samplers."""

from absl.testing import absltest

from fedjax.core import client_datasets
from fedjax.core import client_samplers

import jax
import numpy as np
import numpy.testing as npt


class ClientSamplersTest(absltest.TestCase):

  def assert_clients_equal(self, actual_clients, expected_clients):
    with self.subTest('client_ids'):
      self.assertEqual([cid for cid, _, _ in actual_clients],
                       [cid for cid, _, _ in expected_clients])
    with self.subTest('client_datasets'):
      self.assertEqual([len(cds) for _, cds, _ in actual_clients],
                       [len(cds) for _, cds, _ in expected_clients])
    with self.subTest('client_rngs'):
      npt.assert_array_equal([crng for _, _, crng in actual_clients],
                             [crng for _, _, crng in expected_clients])

  def test_uniform_shuffled_client_sampler(self):

    def shuffled_clients():
      i = 0
      while True:
        yield i, client_datasets.ClientDataset({'x': np.arange(i)})
        i += 1

    with self.subTest('sample'):
      client_sampler1 = client_samplers.UniformShuffledClientSampler(
          shuffled_clients(), num_clients=2)
      for _ in range(4):
        clients1 = client_sampler1.sample()
      client_sampler2 = client_samplers.UniformShuffledClientSampler(
          shuffled_clients(), num_clients=2, start_round_num=3)
      self.assert_clients_equal(client_sampler2.sample(), clients1)

    with self.subTest('set_round_num'):
      with self.assertRaisesRegex(NotImplementedError,
                                  '.*Use UniformGetClientSampler.*'):
        client_sampler1.set_round_num(1)

  def test_uniform_get_client_sampler(self):

    class MockFederatedData:

      def client_ids(self):
        yield from range(100)

      def get_clients(self, client_ids):
        for cid in client_ids:
          yield cid, client_datasets.ClientDataset({'x': np.arange(int(cid))})

    num_clients = 2
    round_num = 3
    client_sampler = client_samplers.UniformGetClientSampler(
        MockFederatedData(), num_clients, seed=0, start_round_num=round_num)
    with self.subTest('sample'):
      client_rngs = jax.random.split(jax.random.PRNGKey(round_num), num_clients)
      expect = [(78, client_datasets.ClientDataset({'x': np.arange(78)}),
                 client_rngs[0]),
                (56, client_datasets.ClientDataset({'x': np.arange(56)}),
                 client_rngs[1])]
      self.assert_clients_equal(client_sampler.sample(), expect)

    with self.subTest('set_round_num'):
      self.assertNotEqual(client_sampler._round_num, round_num)
      client_sampler.set_round_num(round_num)
      self.assert_clients_equal(client_sampler.sample(), expect)


if __name__ == '__main__':
  absltest.main()
