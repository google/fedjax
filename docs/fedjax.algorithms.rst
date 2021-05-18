fedjax.algorithms package
=========================

.. automodule:: fedjax.algorithms

.. autosummary::
      :nosignatures:

      fedjax.algorithms.agnostic_fed_avg
      fedjax.algorithms.fed_avg
      fedjax.algorithms.hyp_cluster
      fedjax.algorithms.mime
      fedjax.algorithms.mime_lite

AgnosticFedAvg
--------------

.. automodule:: fedjax.algorithms.agnostic_fed_avg

.. autoclass:: fedjax.algorithms.agnostic_fed_avg.ServerState

.. autofunction:: fedjax.algorithms.agnostic_fed_avg.agnostic_federated_averaging

FedAvg
------

.. automodule:: fedjax.algorithms.fed_avg

.. autoclass:: fedjax.algorithms.fed_avg.ServerState

.. autofunction:: fedjax.algorithms.fed_avg.federated_averaging

HypCluster
----------

.. automodule:: fedjax.algorithms.hyp_cluster

.. autoclass:: fedjax.algorithms.hyp_cluster.ServerState

.. autofunction:: fedjax.algorithms.hyp_cluster.hyp_cluster

.. autofunction:: fedjax.algorithms.hyp_cluster.random_init

.. autofunction:: fedjax.algorithms.hyp_cluster.kmeans_init

.. autoclass:: fedjax.algorithms.hyp_cluster.ModelKMeansInitializer
  :members:
  :undoc-members:
  :special-members: __init__

.. autoclass:: fedjax.algorithms.hyp_cluster.HypClusterEvaluator
  :members: evaluate_clients
  :special-members: __init__

Mime
----

.. automodule:: fedjax.algorithms.mime

.. autoclass:: fedjax.algorithms.mime.ServerState

.. autofunction:: fedjax.algorithms.mime.mime

MimeLite
--------

.. automodule:: fedjax.algorithms.mime_lite

.. autofunction:: fedjax.algorithms.mime_lite.mime_lite
