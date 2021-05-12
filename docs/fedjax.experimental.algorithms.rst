fedjax.experimental.algorithms package
======================================

.. automodule:: fedjax.experimental.algorithms

.. autosummary::
      :nosignatures:

      fedjax.experimental.algorithms.agnostic_fed_avg
      fedjax.experimental.algorithms.fed_avg
      fedjax.experimental.algorithms.hyp_cluster
      fedjax.experimental.algorithms.mime
      fedjax.experimental.algorithms.mime_lite

AgnosticFedAvg
--------------

.. automodule:: fedjax.experimental.algorithms.agnostic_fed_avg

.. autoclass:: fedjax.experimental.algorithms.agnostic_fed_avg.ServerState

.. autofunction:: fedjax.experimental.algorithms.agnostic_fed_avg.agnostic_federated_averaging

FedAvg
------

.. automodule:: fedjax.experimental.algorithms.fed_avg

.. autoclass:: fedjax.experimental.algorithms.fed_avg.ServerState

.. autofunction:: fedjax.experimental.algorithms.fed_avg.federated_averaging

HypCluster
----------

.. automodule:: fedjax.experimental.algorithms.hyp_cluster

.. autoclass:: fedjax.experimental.algorithms.hyp_cluster.ServerState

.. autofunction:: fedjax.experimental.algorithms.hyp_cluster.hyp_cluster

.. autofunction:: fedjax.experimental.algorithms.hyp_cluster.random_init

.. autofunction:: fedjax.experimental.algorithms.hyp_cluster.kmeans_init

.. autoclass:: fedjax.experimental.algorithms.hyp_cluster.ModelKMeansInitializer
	:members: 
	:undoc-members:
	:special-members: __init__

.. autoclass:: fedjax.experimental.algorithms.hyp_cluster.HypClusterEvaluator
	:members: evaluate_clients
	:special-members: __init__

Mime
----

.. automodule:: fedjax.experimental.algorithms.mime

.. autoclass:: fedjax.experimental.algorithms.mime.ServerState

.. autofunction:: fedjax.experimental.algorithms.mime.mime

MimeLite
--------

.. automodule:: fedjax.experimental.algorithms.mime_lite

.. autofunction:: fedjax.experimental.algorithms.mime_lite.mime_lite
