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
"""Tests for fedjax.legacy.training.federated_experiment."""

import glob
import os.path

from absl.testing import absltest
from fedjax.core import client_datasets
from fedjax.core import client_samplers
from fedjax.core import federated_algorithm
from fedjax.core import in_memory_federated_data
from fedjax.core import metrics
from fedjax.core import models
from fedjax.training import federated_experiment
import jax
import numpy as np
import numpy.testing as npt


class FakeClientSampler(client_samplers.ClientSampler):
  """Sequentially creates single client samples."""

  def __init__(self, base=0):
    self._base = base
    self._round_num = 0

  def set_round_num(self, round_num):
    self._round_num = round_num

  def sample(self):
    client_id = self._round_num + self._base
    dataset = client_datasets.ClientDataset({
        'x': np.array([client_id], dtype=np.int32),
        'y': np.array([client_id], dtype=np.int32) % 2
    })
    rng = None
    self._round_num += 1
    return [(client_id, dataset, rng)]


def fake_algorithm():
  """Counts the number of clients and sums up the 'x' feature."""

  def init():
    # num_clients, sum_values
    return 0, 0

  def apply(state, clients):
    num_clients, sum_values = state
    for _, dataset, _ in clients:
      num_clients += 1
      for x in dataset.all_examples()['x']:
        sum_values += x
    state = num_clients, sum_values
    return state, None

  return federated_algorithm.FederatedAlgorithm(init, apply)


class FakeEvaluationFn(federated_experiment.EvaluationFn):

  def __init__(self, test, expected_state):
    self._test = test
    self._expected_state = expected_state

  def __call__(self, state, round_num):
    self._test.assertEqual(state, self._expected_state[round_num])
    return {'round_num': round_num}


class FakeTrainClientsEvaluationFn(federated_experiment.TrainClientsEvaluationFn
                                  ):

  def __init__(self, test, expected_state, expected_client_ids):
    self._test = test
    self._expected_state = expected_state
    self._expected_client_ids = expected_client_ids

  def __call__(self, state, round_num, train_clients):
    self._test.assertEqual(state, self._expected_state[round_num])
    client_ids = [i[0] for i in train_clients]
    self._test.assertCountEqual(client_ids,
                                self._expected_client_ids[round_num])
    return {'round_num': round_num}


class RunFederatedExperimentTest(absltest.TestCase):

  def test_no_eval(self):
    config = federated_experiment.FederatedExperimentConfig(
        root_dir=self.create_tempdir(), num_rounds=5)
    client_sampler = FakeClientSampler()
    algorithm = fake_algorithm()
    state = federated_experiment.run_federated_experiment(
        algorithm=algorithm,
        init_state=algorithm.init(),
        client_sampler=client_sampler,
        config=config)
    self.assertEqual(state, (5, 15))

  def test_checkpoint(self):
    with self.subTest('checkpoint init'):
      config = federated_experiment.FederatedExperimentConfig(
          root_dir=self.create_tempdir(), num_rounds=5, checkpoint_frequency=3)
      client_sampler = FakeClientSampler()
      algorithm = fake_algorithm()
      state = federated_experiment.run_federated_experiment(
          algorithm=algorithm,
          init_state=(1, -1),
          client_sampler=client_sampler,
          config=config)
      self.assertEqual(state, (6, 14))
      self.assertCountEqual(
          glob.glob(os.path.join(config.root_dir, 'checkpoint_*')),
          [os.path.join(config.root_dir, 'checkpoint_00000003')])

    with self.subTest('checkpoint restore'):
      # Restored state is (4, 5). FakeSampler produces clients [5, 6].
      state = federated_experiment.run_federated_experiment(
          algorithm=algorithm,
          init_state=None,
          client_sampler=FakeClientSampler(1),
          config=config)
      self.assertEqual(state, (6, 16))
      self.assertCountEqual(
          glob.glob(os.path.join(config.root_dir, 'checkpoint_*')),
          [os.path.join(config.root_dir, 'checkpoint_00000004')])

  def test_periodic_eval_fn_map(self):
    config = federated_experiment.FederatedExperimentConfig(
        root_dir=self.create_tempdir(), num_rounds=5, eval_frequency=3)
    client_sampler = FakeClientSampler()
    algorithm = fake_algorithm()
    expected_state = {1: (1, 1), 3: (3, 6)}
    expected_client_ids = {1: [1], 3: [3]}
    state = federated_experiment.run_federated_experiment(
        algorithm=algorithm,
        init_state=algorithm.init(),
        client_sampler=client_sampler,
        config=config,
        periodic_eval_fn_map={
            'test_eval':
                FakeEvaluationFn(self, expected_state),
            'train_eval':
                FakeTrainClientsEvaluationFn(self, expected_state,
                                             expected_client_ids)
        })
    self.assertEqual(state, (5, 15))
    self.assertCountEqual(
        glob.glob(os.path.join(config.root_dir, '*eval*')), [
            os.path.join(config.root_dir, 'test_eval'),
            os.path.join(config.root_dir, 'train_eval')
        ])

  def test_final_eval_fn_map(self):
    config = federated_experiment.FederatedExperimentConfig(
        root_dir=self.create_tempdir(), num_rounds=5, eval_frequency=3)
    client_sampler = FakeClientSampler()
    algorithm = fake_algorithm()
    expected_state = {5: (5, 15)}
    state = federated_experiment.run_federated_experiment(
        algorithm=algorithm,
        init_state=algorithm.init(),
        client_sampler=client_sampler,
        config=config,
        final_eval_fn_map={
            'final_eval': FakeEvaluationFn(self, expected_state)
        })
    self.assertEqual(state, (5, 15))
    self.assertCountEqual(
        glob.glob(os.path.join(config.root_dir, 'final_eval.tsv')),
        [os.path.join(config.root_dir, 'final_eval.tsv')])


def fake_model():

  def apply_for_eval(params, example):
    del params
    return jax.nn.one_hot(example['x'] % 3, 3)

  eval_metrics = {'accuracy': metrics.Accuracy()}
  return models.Model(
      init=None,
      apply_for_train=None,
      apply_for_eval=apply_for_eval,
      train_loss=None,
      eval_metrics=eval_metrics)


class FakeState:
  params = None


class EvaluationFnsTest(absltest.TestCase):

  def test_model_sample_clients_evaluation_fn(self):
    eval_fn = federated_experiment.ModelSampleClientsEvaluationFn(
        FakeClientSampler(), fake_model(),
        client_datasets.PaddedBatchHParams(batch_size=4))
    state = FakeState()
    npt.assert_equal(eval_fn(state, 1), {'accuracy': np.array(1.)})
    npt.assert_equal(eval_fn(state, 2), {'accuracy': np.array(0.)})
    npt.assert_equal(eval_fn(state, 3), {'accuracy': np.array(0.)})
    npt.assert_equal(eval_fn(state, 4), {'accuracy': np.array(0.)})
    npt.assert_equal(eval_fn(state, 5), {'accuracy': np.array(0.)})
    npt.assert_equal(eval_fn(state, 6), {'accuracy': np.array(1.)})

  def test_model_full_evaluation_fn(self):
    sampler = FakeClientSampler()
    sampler.set_round_num(1)
    clients = [sampler.sample()[0] for _ in range(4)]
    fd = in_memory_federated_data.InMemoryFederatedData(
        dict((k, v.all_examples()) for k, v, _ in clients))
    eval_fn = federated_experiment.ModelFullEvaluationFn(
        fd, fake_model(), client_datasets.PaddedBatchHParams(batch_size=4))
    state = FakeState()
    npt.assert_equal(eval_fn(state, 1), {'accuracy': np.array(0.25)})
    npt.assert_equal(eval_fn(state, 100), {'accuracy': np.array(0.25)})

  def test_model_train_clients_evaluation_fn(self):
    sampler = FakeClientSampler()
    sampler.set_round_num(1)
    clients = [sampler.sample()[0] for _ in range(4)]
    eval_fn = federated_experiment.ModelTrainClientsEvaluationFn(
        fake_model(), client_datasets.PaddedBatchHParams(batch_size=4))
    state = FakeState()
    npt.assert_equal(eval_fn(state, 1, clients), {'accuracy': np.array(0.25)})
    npt.assert_equal(eval_fn(state, 100, clients), {'accuracy': np.array(0.25)})


if __name__ == '__main__':
  absltest.main()
