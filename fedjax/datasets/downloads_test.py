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
"""Tests for fedjax.datasets.downloads.

Can only be run locally because of network access.
"""

import os
import os.path
import shutil

from absl import flags
from absl.testing import absltest
from fedjax.datasets import downloads

import sqlite3

FLAGS = flags.FLAGS


class DownloadsTest(absltest.TestCase):
  SOURCE = 'https://storage.googleapis.com/tff-datasets-public/shakespeare.sqlite.lzma'
  HEXDIGEST = 'd3d11fceb9e105439ac6f4d52af6efafed5a2a1e1eb24c5bd2dd54ced242f5c4'
  NUM_BYTES = 1329828

  def setUp(self):
    super().setUp()
    # This will be set in actual tests.
    self._cache_dir = None

  def tearDown(self):
    super().tearDown()
    if self._cache_dir is not None:
      shutil.rmtree(self._cache_dir)

  def _with_cache_dir(self, cache_dir, expected_path):
    # First access, being copied.
    path = downloads.maybe_download(self.SOURCE, cache_dir)
    self._cache_dir = os.path.dirname(path)
    self.assertEqual(path, expected_path)
    downloads.validate_file(path, self.NUM_BYTES, self.HEXDIGEST)
    stat = os.stat(path)
    # ctime >= mtime because of renaming.
    self.assertGreaterEqual(stat.st_ctime_ns, stat.st_mtime_ns)

    # Second access, no copy.
    path1 = downloads.maybe_download(self.SOURCE, cache_dir)
    self.assertEqual(path1, path)
    downloads.validate_file(path1, self.NUM_BYTES, self.HEXDIGEST)
    # Check that the file hasn't been overwritten.
    stat1 = os.stat(path1)
    self.assertEqual(stat1.st_mtime_ns, stat.st_mtime_ns)

  def test_default(self):
    self._with_cache_dir(
        cache_dir=None,
        expected_path=os.path.join(
            os.path.expanduser('~'), '.cache/fedjax/shakespeare.sqlite.lzma'))

  def test_cache_dir(self):
    cache_dir = os.path.join(FLAGS.test_tmpdir, 'test_cache_dir')
    cache_file = os.path.join(cache_dir, 'shakespeare.sqlite.lzma')
    self._with_cache_dir(cache_dir=cache_dir, expected_path=cache_file)

  def test_partial_file(self):
    cache_dir = os.path.join(FLAGS.test_tmpdir, 'test_partial_file')
    cache_file = os.path.join(cache_dir, 'shakespeare.sqlite.lzma')
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file + '.partial', 'w') as f:
      f.write('hel')
    self._with_cache_dir(cache_dir=cache_dir, expected_path=cache_file)

  def test_maybe_lzma_decompress(self):
    cache_dir = os.path.join(FLAGS.test_tmpdir, 'test_maybe_lzma_decompress')
    compress_path = downloads.maybe_download(self.SOURCE, cache_dir)
    decompress_path = downloads.maybe_lzma_decompress(compress_path)
    connection = sqlite3.connect(decompress_path)
    with connection:
      cursor = connection.execute(
          'SELECT COUNT(*) FROM client_metadata WHERE split_name = "train";')
      self.assertEqual(cursor.fetchone()[0], 715)
    connection.close()


if __name__ == '__main__':
  absltest.main()
