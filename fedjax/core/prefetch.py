"""Utilities for prefetching from tff.simulation.ClientData."""

import collections
from concurrent import futures
import functools
from typing import Callable, Iterable, Optional

from fedjax.core import tree_util
from fedjax.core import typing
import tensorflow as tf

DEFAULT_ASYNC_TF_DATASET_ITERATOR_PREFETCH = 2
DEFAULT_PREFETCH_THREAD_POOL_SIZE = 10
DEFAULT_PREFETCH_NUM_INIT_FETCH = 10


class AsyncTFDatasetIterator:
  """Asynchronously create a TF dataset iterator, and fetches the first n values.

  TensorFlow dataset iterators are expensive to create, and first values out of
  the iteration often take more time than the rest. We thus wrap these
  computation into a future, so that we can submit it well before we need the
  data.

  While FutureIterator can generalize to other types of iterables, they are
  useless for pure Python iterators because of GIL. Thus we only handle
  TensorFlow datasets.
  """

  def __init__(self,
               executor: futures.ThreadPoolExecutor,
               dataset_fn: Callable[[], tf.data.Dataset],
               num_prefetch: Optional[int] = None):
    """Initializes the iterator.

    Args:
      executor: Executor to submit future computation to.
      dataset_fn: Callable that creates the underlying TensorFlow dataset.
      num_prefetch: Number of values to fetch during the future call. Defaults
        to DEFAULT_ASYNC_TF_DATASET_ITERATOR_PREFETCH.
    """
    if num_prefetch is None:
      num_prefetch = DEFAULT_ASYNC_TF_DATASET_ITERATOR_PREFETCH

    def prefetch():
      """Creates the iterator and fetches the first few values."""
      iterator = dataset_fn().as_numpy_iterator()
      values = collections.deque()
      for _ in range(num_prefetch):
        try:
          v = next(iterator)
        except StopIteration:
          break
        values.append(v)
      return values, iterator

    self._pending = executor.submit(prefetch)
    self._values, self._iterator = None, None

  def __iter__(self):
    return self

  def __next__(self):
    if self._pending is not None:
      # Calling result() will also raise any uncaught exceptions in prefetch().
      # Therefore we don't need any special handling like we do for
      # PrefetchClientDatasetsIterator.
      self._values, self._iterator = self._pending.result()
      self._pending = None
    if self._values:
      return self._values.popleft()
    return next(self._iterator)


class PrefetchClientDatasetsIterator:
  """Prefetches per-client datasets from one or more fedjax.FederatedData.

  This class uses asynchrounous prefetching to hide per-client dataset
  creation/iteration overhead. At each `next()` call,
  PrefetchClientDatasetsIterator produces one iterator for each
  fedjax.FederatedData in the PyTree given during `__init__()`. These iterators
  can be used to iterate over the corresponding client dataset.

  Example usage:
    federated_data = (train_data, (eval_train_data, eval_test_data))
    client_ids = ["0", "1", ...]
    data_iterator = PrefetchClientDatasetsIterator(federated_data, client_ids)
    # Iterators for client_id "0"
    client_id_0, iters_0 = next(data_iterator)
    train_iter_0, (eval_train_iter_0, eval_test_iter_0) = iters_0
    # Iterators for client_id "1"
    client_id_1, iters_1 = next(data_iterator)
    train_iter_1, (eval_train_iter_1, eval_test_iter_1) = iters_1
  """

  def __init__(self,
               federated_data: typing.PyTree,
               client_ids: Iterable[str],
               num_threads: Optional[int] = None,
               num_init_fetch: Optional[int] = None):
    """Initializes a PrefetchClientDatasetsIterator.

    Args:
      federated_data: PyTree of fedjax.FederatedData.
      client_ids: Client ids in the order the caller expects to access.
      num_threads: Number of prefetching threads. Defaults to
        DEFAULT_PREFETCH_THREAD_POOL_SIZE.
      num_init_fetch: Number of initial fetches to schedule during __init__().
        Defaults to DEFAULT_PREFETCH_NUM_INIT_FETCH.
    """
    if num_threads is None:
      num_threads = DEFAULT_PREFETCH_THREAD_POOL_SIZE
    if num_init_fetch is None:
      num_init_fetch = DEFAULT_PREFETCH_NUM_INIT_FETCH
    self._federated_data = federated_data
    self._client_ids = iter(client_ids)
    self._executor = futures.ThreadPoolExecutor(num_threads)
    # Each value is a (value, exception) tuple.
    self._buf = collections.deque()
    self._done = False
    for _ in range(num_init_fetch):
      self._fetch()

  def __del__(self):
    self._executor.shutdown(wait=False)

  def __iter__(self):
    return self

  def __next__(self):
    v, e = self._buf.popleft()
    if e is not None:
      raise e
    self._fetch()
    return v

  def _fetch(self):
    """Fetches the iterators for the next client."""
    if self._done:
      return
    try:
      client_id = next(self._client_ids)
    except StopIteration as e:
      self._buf.append((None, e))
      self._done = True
      return

    def create_iterator(one_federated_data):
      return AsyncTFDatasetIterator(
          self._executor,
          functools.partial(one_federated_data.create_tf_dataset_for_client,
                            client_id))

    try:
      v = (client_id, tree_util.tree_map(create_iterator, self._federated_data))
      self._buf.append((v, None))
    except Exception as e:  # pylint: disable=broad-except
      self._buf.append((None, e))
