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
"""Simple download manager."""

import hashlib
import math
import os.path
import shutil
import sys
import time
from typing import Callable, Iterator, Optional
import urllib.parse

import lzma
import requests


def progress(n: int) -> Iterator[int]:
  """A simple generator for tracking progress."""
  start = time.time()
  last_log = -1
  for i in range(n):
    yield i
    elapsed = time.time() - start
    if elapsed - last_log >= 1:
      last_log = elapsed
      eta = elapsed / (i + 1) * (n - i - 1)
      log(f'\r{100*(i + 1)/n:3.0f}%, ETA: {format_duration(eta)}', end='')
  log(f'\r100%, elapsed: {format_duration(time.time() - start)}')


def maybe_download(url: str,
                   cache_dir: Optional[str] = None,
                   progress_: Callable[[int], Iterator[int]] = progress) -> str:
  """Downloads `url` to local disk.

  Args:
    url: URL to download from.
    cache_dir: Where to cache the file. If None, uses default_cache_dir().
    progress_: A callable that yields like range(n), for tracking progress.

  Returns:
    Path to local file.
  """
  # TODO(wuke): Avoid race conditions when downloading the same file from
  # different threads/processes at the same time.
  if cache_dir is None:
    cache_dir = default_cache_dir()
  os.makedirs(cache_dir, exist_ok=True)
  path = os.path.join(cache_dir,
                      os.path.basename(urllib.parse.urlparse(url).path))
  if os.path.exists(path):
    log(f'Reusing cached file {path!r}')
  else:
    log(f'Downloading {url!r} to {path!r}')
    with open(path + '.partial', 'wb') as fo:
      r = requests.get(url, stream=True)
      r.raise_for_status()
      length = int(r.headers['content-length'])
      block_size = 1 << 18
      for _ in progress_((length + block_size - 1) // block_size):
        fo.write(r.raw.read(block_size))
    os.rename(path + '.partial', path)
  return path


def format_duration(seconds: float) -> str:
  """Formats duration in seconds into hours/minutes/seconds."""
  if seconds < 60:
    return f'{seconds:.0f}s'
  elif seconds < 3600:
    minutes = math.floor(seconds / 60)
    seconds -= minutes * 60
    return f'{minutes}m{seconds:.0f}s'
  else:
    hours = math.floor(seconds / 3600)
    seconds -= hours * 3600
    minutes = math.floor(seconds / 60)
    seconds -= minutes * 60
    return f'{hours}h{minutes}m{seconds:.0f}s'


def log(*args, **kwargs):
  print(*args, flush=True, file=sys.stderr, **kwargs)


def default_cache_dir() -> str:
  """Returns default local file cache directory."""
  running_on_colab = 'google.colab' in sys.modules
  if running_on_colab:
    base_dir = '/tmp'
  else:
    base_dir = os.path.expanduser('~')
  cache_dir = os.path.join(base_dir, '.cache/fedjax')
  return cache_dir


def maybe_lzma_decompress(path) -> str:
  """Decompresses LZMA compressed local file or reuses cached file."""
  decompressed_path, ext = os.path.splitext(path)
  if ext != '.lzma':
    raise ValueError(
        'Only decompressing LZMA files is supported. If the file '
        'is LZMA compressed, rename the url to have a .lzma suffix.')
  if os.path.exists(decompressed_path):
    log(f'Reusing cached file {decompressed_path!r}')
  else:
    log(f'Decompressing {path!r} to {decompressed_path!r}')
    with lzma.open(path, 'rb') as fi:
      with open(decompressed_path, 'wb') as fo:
        shutil.copyfileobj(fi, fo)
  return decompressed_path


def validate_file(path: str, expected_num_bytes: int, expected_hexdigest: str):
  """Validates path file contents has specified number of bytes and hash."""
  with open(path, 'rb') as f:
    data = f.read()
  num_bytes = len(data)
  if num_bytes != expected_num_bytes:
    raise ValueError(
        f'Expected file content number of bytes to be {expected_num_bytes} but found {num_bytes}.'
    )
  hexdigest = hashlib.sha256(data).hexdigest()
  if hexdigest != expected_hexdigest:
    raise ValueError(
        f'Expected file content hash to be {expected_hexdigest!r} but found {hexdigest!r}.'
    )
