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
"""Tests for fedjax.core.dataset_util."""

import collections

from absl.testing import parameterized
from fedjax.core import dataset_util
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff


class DataTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='shuffle',
          hparams=dataset_util.ClientDataHParams(
              batch_size=3, num_epochs=1, shuffle_buffer_size=1),
          expected_num_batches=4),
      dict(
          testcase_name='num_epochs',
          hparams=dataset_util.ClientDataHParams(batch_size=3, num_epochs=5),
          expected_num_batches=17),
      dict(
          testcase_name='drop_remainder',
          hparams=dataset_util.ClientDataHParams(
              batch_size=3, num_epochs=2, drop_remainder=True),
          expected_num_batches=6),
      dict(
          testcase_name='keep_remainder',
          hparams=dataset_util.ClientDataHParams(
              batch_size=3, num_epochs=2, drop_remainder=False),
          expected_num_batches=7),
      dict(
          testcase_name='num_batches',
          hparams=dataset_util.ClientDataHParams(
              batch_size=3, num_epochs=2, num_batches=5),
          expected_num_batches=5),
  )
  def test_preprocess_tf_dataset(self, hparams, expected_num_batches):
    x = np.arange(10 * 2).reshape((10, 2))
    numpy_data = collections.OrderedDict(x=x, y=x, z=x)
    dataset = tf.data.Dataset.from_tensor_slices(numpy_data)

    batches = list(dataset_util.preprocess_tf_dataset(dataset, hparams))

    self.assertLen(batches, expected_num_batches)

  @parameterized.named_parameters(
      dict(
          testcase_name='select_all',
          all_client_ids=None,
          select_client_ids=None,
          expected_data=[
              collections.OrderedDict(x=[1, 2], y=[2, 4]),
              collections.OrderedDict(x=[3, 4], y=[6, 8]),
              collections.OrderedDict(x=[5, 6], y=[10, 12]),
              collections.OrderedDict(x=[7, 8], y=[14, 16]),
              collections.OrderedDict(x=[9, 10], y=[18, 20]),
          ]),
      dict(
          testcase_name='select_subset',
          all_client_ids=None,
          select_client_ids=['a', 'c'],
          expected_data=[
              collections.OrderedDict(x=[1, 2], y=[2, 4]),
              collections.OrderedDict(x=[3, 4], y=[6, 8]),
              collections.OrderedDict(x=[9, 10], y=[18, 20]),
          ]))
  def test_create_tf_dataset_for_clients(self, all_client_ids,
                                         select_client_ids, expected_data):
    tensor_slices_dict = collections.OrderedDict(
        a=collections.OrderedDict(x=[[1, 2], [3, 4]], y=[[2, 4], [6, 8]]),
        b=collections.OrderedDict(x=[[5, 6], [7, 8]], y=[[10, 12], [14, 16]]),
        c=collections.OrderedDict(x=[[9, 10]], y=[[18, 20]]),
    )
    mock_federated_data = tff.test.FromTensorSlicesClientData(
        tensor_slices_dict)

    actual_dataset = dataset_util.create_tf_dataset_for_clients(
        mock_federated_data, select_client_ids)
    actual_data = tf.nest.map_structure(lambda x: x.numpy().tolist(),
                                        list(actual_dataset))

    actual_data = sorted(actual_data, key=lambda d: d['x'][0])
    expected_data = sorted(expected_data, key=lambda d: d['x'][0])
    self.assertSameStructure(actual_data, expected_data)

  def test_dataset_or_iterable(self):
    slices = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # Access through TF dataset.
    for i, x in enumerate(
        dataset_util.iterate(tf.data.Dataset.from_tensor_slices(slices))):
      self.assertIsInstance(x, np.ndarray, msg=type(x))
      self.assertAllEqual(x, slices[i])
    # Access through direct iteration.
    for i, x in enumerate(dataset_util.iterate(slices)):
      self.assertIsInstance(x, np.ndarray, msg=type(x))
      self.assertAllEqual(x, slices[i])


if __name__ == '__main__':
  tf.test.main()
