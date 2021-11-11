fedjax.algorithms
=================

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
  :members: ServerState, agnostic_federated_averaging

FedAvg
------

.. automodule:: fedjax.algorithms.fed_avg
  :members: ServerState, federated_averaging

HypCluster
----------

.. automodule:: fedjax.algorithms.hyp_cluster
  :members: ServerState, hyp_cluster, random_init, kmeans_init

.. autoclass:: ModelKMeansInitializer
  :members:
  :undoc-members:
  :special-members: __init__

.. autoclass:: fedjax.algorithms.hyp_cluster.HypClusterEvaluator
  :members:
  :undoc-members:
  :special-members: __init__

Mime
----

.. automodule:: fedjax.algorithms.mime
  :members: ServerState, mime

MimeLite
--------

.. automodule:: fedjax.algorithms.mime_lite
  :members: mime_lite
