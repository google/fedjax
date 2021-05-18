fedjax.metrics module
=====================

.. automodule:: fedjax.metrics

Stats
-----

.. autosummary::
    :nosignatures:

    fedjax.metrics.Stat
    fedjax.metrics.MeanStat
    fedjax.metrics.SumStat

Metrics
-------

.. autosummary::
    :nosignatures:

    fedjax.metrics.Metric
    fedjax.metrics.CrossEntropyLoss
    fedjax.metrics.Accuracy
    fedjax.metrics.SequenceTokenCrossEntropyLoss
    fedjax.metrics.SequenceCrossEntropyLoss
    fedjax.metrics.SequenceTokenAccuracy
    fedjax.metrics.SequenceTokenCount
    fedjax.metrics.SequenceCount
    fedjax.metrics.SequenceTruncationRate
    fedjax.metrics.SequenceTokenOOVRate
    fedjax.metrics.SequenceLength
    fedjax.metrics.PerDomainMetric

Miscellaneous
-------------

.. autosummary::

    fedjax.metrics.unreduced_cross_entropy_loss
    fedjax.metrics.evaluate_batch

Quick Overview
--------------

To evaluate model predictions, use a :class:`Metric` object such as :class:`Accuracy` .
We recommend :func:`fedjax.core.models.evaluate_model` in most scenarios,
which runs model prediction, and evaluation, on batches of N examples at a time
for greater computational efficiency::

  # Mock out Model.
  model = fedjax.Model.new(
      init=lambda _: None,  # Unused.
      apply_for_train=lambda _, _, _: None,  # Unused.
      apply_for_eval=lambda _, batch: batch.get('pred'),
      train_loss=lambda _, _: None,  # Unused.
      eval_metrics={'accuracy': metrics.Accuracy()})
  params = None  # Unused.
  batches = [{'y': np.array([1, 0]),
              'pred': np.array([[1.2, 0.4], [2.3, 0.1]])},
             {'y': np.array([1, 1]),
              'pred': np.array([[0.3, 3.2], [2.1, 4.3]])}]
  results = fedjax.evaluate_model(model, params, batches)
  print(results)
  # {'accuracy': 0.75}

A :class:`Metric` object has 2 methods:

* :meth:`~Metric.zero` : Initial value for accumulating the statistic for this metric.
* :meth:`~Metric.evaluate_example` : Returns the statistic from evaluating a single example, given the training ``example`` and the model ``prediction``.

Most :class:`Metric` follow the following convention for convenience:

* ``example`` is a dict-like object from ``str`` to ``jnp.ndarray``.
* ``prediction`` is either a single ``jnp.ndarray``, or a dict-like object from ``str`` to ``jnp.ndarray``.

Conceptually, we can also use a simple for loop to evaluate a collection of
examples and model predictions::

  # By default, the `Accuracy` metric treats `example['y']` as the true label,
  # and `prediction` as a single `jnp.ndarray` of class scores.
  metric = Accuracy()
  stat = metric.zero()
  # We are iterating over individual examples, not batches.
  for example, prediction in [({'y': jnp.array(1)}, jnp.array([0., 1.])),
                              ({'y': jnp.array(0)}, jnp.array([1., 0.])),
                              ({'y': jnp.array(1)}, jnp.array([1., 0.])),
                              ({'y': jnp.array(0)}, jnp.array([2., 0.]))]:
    stat = stat.merge(metric.evaluate_example(example, prediction))
  print(stat.result())
  # 0.75

In practice, for greater computational efficiency, we run model prediction not
on a single example, but a batch of N examples at a time.
:func:`fedjax.core.models.evaluate_model` provides a simple way to do so.
Under the hood, it calls :func:`evaluate_batch` ::

  metric = Accuracy()
  stat = metric.zero()
  # We are iterating over batches.
  for batch_example, batch_prediction in [
    ({'y': jnp.array([1, 0])}, jnp.array([[0., 1.], [1., 0.]])),
    ({'y': jnp.array([1, 0])}, jnp.array([[1., 0.], [2., 0.]]))]:
    stat = stat.merge(evaluate_batch(metric, batch_example, batch_prediction))
  print(stat.result())
  # 0.75

.. _under-the-hood:

Under the hood
--------------

For most users, it is sufficient to know how to use existing :class:`Metric`
subclasses such as :class:`Accuracy` with
:func:`fedjax.core.models.evaluate_model` .
This section is intended for those who would like to write new metrics.

From algebraic structures to :class:`Metric` and :class:`Stat`
##############################################################

There are 2 abstraction in this library, :class:`Metric` and :class:`Stat` .
Before going into details of these classes, let's first consider a few abstract
properties related to evaluation metrics, using accuracy as an example.

When evaluating accuracy on a dataset, we wish to know the proportion of
examples that are correctly predicted by the model. Because a dataset might be
too large to fit into memory, we need to divide the work by partitioning the
dataset into smaller subsets, evaluate each separately, and finally somehow
combine the results. Assuming the subsets can be of different sizes, although
the accuracy value is a single number, we cannot just average the accuracy
values from each partition to arrive at the overall accuracy. Instead, we need 2
numbers from each subset:

* The number of examples in this subset,
* How many of them are correctly predicted.

We call these two numbers from each subset a *statistic*. The domain (the set of
possible values) of the statistic in the case of accuracy is

.. math::

   \{(0, 0)\} ∪ \{(a, b) | a >= 0, b > 0\}

With the numbers of examples and correct predictions from 2 disjoint subsets, we
add the numbers up to get the number of examples and correct predictions for the
union of the 2 subsets. We call this operation from 2 statistics into 1 a
:math:`merge` operation.

Let :math:`f(S)` be the function that gives us the statistic from a subset of
examples. It is easy to see for two disjoint subsets :math:`A` and :math:`B` ,
:math:`merge(f(A), f(B))` should be equal to :math:`f(A ∪ B)` . If no such
:math:`merge` exists, we cannot evaluate the dataset by partitioning the work.
This requirement alone implies the domain of a statistic, and the :math:`merge`
operation forms a specific algebraic structure (a commutative monoid).

* | :math:`I := f(empty set)` is one and the only identity element w.r.t.
  | :math:`merge` (i.e. :math:`merge(I, x) == merge(x, I) == x` ).
* :math:`merge()` is commutative and associative.

Further, we can see :math:`f(S)` can be defined just knowing two types of values:

* :math:`f(empty set)`, i.e. :math:`I` ;
* :math:`f({x})` for any single example :math:`x` .

For any other subset :math:`S` , we can derive the value of :math:`f(S)` using
these values and :math:`merge` . :class:`Metric` is simply the :math:`f(S)` function
above, defined in 2 corresponding parts:

* :meth:`Metric.zero` is :math:`f(empty set)` .
* :meth:`Metric.evaluate_example` is :math:`f({x})` for a single example.

On the other hand, :class:`Stat` stores a single statistic, a :meth:`~Stat.merge`
method for combining two, and a :meth:`~Stat.result` method for producing the
final metric value.

To implement :class:`Accuracy` as a subclass of :class:`Metric` , we first need to
know what :class:`Stat` to use. In this case, the statistic domain and merge is
implemented by a :class:`MeanStat` . A :class:`MeanStat` holds two values:

* | :attr:`~MeanStat.accum` is the weighted sum of values, i.e. the number of correct
  | predictions in the case of accuracy.
* | :attr:`~MeanStat.weight` is the sum of weights, i.e. the number of examples in
  | the case of accuracy.

:meth:`~MeanStat.merge` adds up the respective :attr:`~MeanStat.accum` and
:attr:`~MeanStat.weight` from two :class:`MeanStat` objects.

Sometimes, a new :class:`Stat` subclass is necessary. In that case, it is very
important to make sure the implementation has a clear definition of the domain,
and the :meth:`~Stat.merge` operation adheres to the properties regarding identity
element, commutativity, and associativity (e.g. if we unknowingly allow pairs of
:math:`(x, 0)` for :math:`x != 0` into the domain of a :class:`MeanStat` ,
:math:`merge((x, 0), (a, b))` will produce a statistic that leads to incorrect
final metric values, i.e. :math:`(a+x)/b` , instead of :math:`a/b` ).

.. _batching-stats:

Batching :class:`Stat` s
########################

In most cases, the final value of an evaluation is simply a scalar, and the
corresponding statistic is also a tuple of a few scalar values. However, for the
same reason why :func:`jax.vmap` is a lot more efficient than a for loop, it is a
lot more efficient to store multiple :class:`Stat` values as a :class:`Stat` of arrays,
instead of a list of :class:`Stat` objects. Thus instead of a list
:code:`[MeanStat(1, 2), MeanStat(3, 4), MeanStat(5, 6)]`
(call these "rank 0" :class:`Stat` s), we can batch the 3 statitstics as a single
``MeanStat(jnp.array([1, 3, 5]), jnp.array([2, 4, 6]))``.
A :class:`Stat` object holding a single statistic is a "rank 0" :class:`Stat` .
A :class:`Stat` object holding a vector of statistics is a "rank 1" :class:`Stat` .
Similarly, a :class:`Stat` object may also hold a matrix, a 3D array, etc, of statistics.
These are higher rank :class:`Stat` s. In most cases, the rank 1 implementation of
:meth:`~Stat.merge` and :meth`~Stat.result` automatically generalizes to higher ranks as
elementwise operations.

In the end, we want just 1 final metric value instead of a length 3 vector, or a
2x2 matrix, of metric values. To finally go back to a single statistic (), we
need to :meth:`~Stat.merge` statistics stored in these arrays. Each :class:`Stat` subclass
provides a :meth:`~Stat.reduce` method to do just that. The combination of :func:`jax.vmap`
over :meth:`Metric.evaluate_example`, and :meth:`Stat.reduce` , is how we get an
efficient :func:`evaluate_batch` function (of course, the real :func:`evaluate_batch`
is :func:`jax.jit` 'd so that the same :func:`jax.vmap` transformation etc does not
need to happen over and over.::

  metric = Accuracy()
  stat = metric.zero()
  # We are iterating over batches.
  for batch_example, batch_prediction in [
    ({'y': jnp.array([1, 0])}, jnp.array([[0., 1.], [1., 0.]])),
    ({'y': jnp.array([1, 0])}, jnp.array([[1., 0.], [2., 0.]]))]:
    # Get a batch of statistics as a single Stat object.
    batch_stat = jax.vmap(metric.evaluate_example)(batch_example,
    batch_prediction)
    # Merge the reduced single statistic onto the accumulator.
    stat = stat.merge(batch_stat.reduce())
  print(stat.result())
  # 0.75


Being able to batch :class:`Stat` s also allows us to do other interesting things, for
example,

* | :func:`evaluate_batch` accepts an optional per-example ``mask`` so it can work
  | on padded batches.
* | We can define a :class:`PerDomainMetric` metric for any base metric so that we can get
  | accuracy where examples are partitioned by a domain id.

Creating a new :class:`Metric`
##############################

Most likely, a new metric will just return a :class:`MeanStat` or a :class:`SumStat`.
If that's the case, simply follow the guidelines in :class:`Metric` 's class docstring.

If a new :class:`Stat` is necessary, follow the guidelines in :class:`Stat` 's docstring.

----

.. autoclass:: fedjax.metrics.Stat
    :members:

.. autoclass:: fedjax.metrics.MeanStat
    :members: merge, new, reduce, result
    :show-inheritance:

.. autoclass:: fedjax.metrics.SumStat
    :members: merge, new, reduce, result
    :show-inheritance:

.. autoclass:: fedjax.metrics.Metric
    :members:

.. autoclass:: fedjax.metrics.CrossEntropyLoss
    :members: evaluate_example, zero
    :undoc-members:
    :show-inheritance:

.. autoclass:: fedjax.metrics.Accuracy
    :members: evaluate_example, zero
    :undoc-members:
    :show-inheritance:

.. autoclass:: fedjax.metrics.SequenceTokenCrossEntropyLoss
    :members: evaluate_example, zero
    :undoc-members:
    :show-inheritance:

.. autoclass:: fedjax.metrics.SequenceCrossEntropyLoss
    :members: evaluate_example, zero
    :undoc-members:
    :show-inheritance:

.. autoclass:: fedjax.metrics.SequenceTokenAccuracy
    :members: evaluate_example, zero
    :undoc-members:
    :show-inheritance:

.. autoclass:: fedjax.metrics.SequenceTokenCount
    :members: evaluate_example, zero
    :undoc-members:
    :show-inheritance:

.. autoclass:: fedjax.metrics.SequenceCount
    :members: evaluate_example, zero
    :undoc-members:
    :show-inheritance:

.. autoclass:: fedjax.metrics.SequenceTruncationRate
    :members: evaluate_example, zero
    :undoc-members:
    :show-inheritance:

.. autoclass:: fedjax.metrics.SequenceTokenOOVRate
    :members: evaluate_example, zero
    :undoc-members:
    :show-inheritance:

.. autoclass:: fedjax.metrics.SequenceLength
    :members: evaluate_example, zero
    :undoc-members:
    :show-inheritance:

.. autoclass:: fedjax.metrics.PerDomainMetric
    :members: evaluate_example, zero
    :undoc-members:
    :show-inheritance:

.. autofunction:: fedjax.metrics.unreduced_cross_entropy_loss

.. autofunction:: fedjax.metrics.evaluate_batch

