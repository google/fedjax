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
"""Tests for fedjax.training.logging."""

import tensorflow as tf

from fedjax.training import logging


class LoggingTest(tf.test.TestCase):

  def test_log_no_root_dir(self):
    logger = logging.Logger()

    logger.log(
        writer_name='train', metric_name='loss', metric_value=4., round_num=0)

    self.assertEmpty(logger._summary_writers)

  def test_log_root_dir(self):
    root_dir = self.create_tempdir()
    logger = logging.Logger(root_dir)

    logger.log(
        writer_name='train', metric_name='loss', metric_value=4.1, round_num=0)
    logger.log(
        writer_name='eval', metric_name='loss', metric_value=5.3, round_num=0)

    self.assertCountEqual(['train', 'eval'], tf.io.gfile.listdir(root_dir))


if __name__ == '__main__':
  tf.test.main()
