fedjax package
==============

.. automodule:: fedjax

Subpackages
-----------

.. toctree::
   :maxdepth: 1		

   fedjax.metrics
   fedjax.optimizers
   fedjax.tree_util

Federated data
--------------

.. autosummary::
    :nosignatures:

    fedjax.FederatedData
    fedjax.ClientPreprocessor
    fedjax.shuffle_repeat_batch_federated_data
    fedjax.padded_batch_federated_data
    fedjax.ClientDataset
    fedjax.BatchPreprocessor
    fedjax.buffered_shuffle_batch_client_datasets
    fedjax.padded_batch_client_datasets

Model
-----

.. autosummary::
    :nosignatures:

    fedjax.Model
    fedjax.create_model_from_haiku
    fedjax.create_model_from_stax
    fedjax.evaluate_model

Federated algorithm
-------------------

.. autosummary::
    :nosignatures:

    fedjax.FederatedAlgorithm

For each client
---------------

.. autosummary::
    :nosignatures:

    fedjax.for_each_client
    fedjax.for_each_client_backend
    fedjax.set_for_each_client_backend
    fedjax.ForEachClientBackend

----

.. autoclass:: fedjax.FederatedData
	:members:
.. autoclass:: fedjax.ClientPreprocessor
	:members:
.. autofunction:: fedjax.shuffle_repeat_batch_federated_data
.. autofunction:: fedjax.padded_batch_federated_data

----

.. autoclass:: fedjax.ClientDataset
	:members:
.. autoclass:: fedjax.BatchPreprocessor
	:members:
.. autofunction:: fedjax.buffered_shuffle_batch_client_datasets
.. autofunction:: fedjax.padded_batch_client_datasets

----

.. autoclass:: fedjax.Model
.. autofunction:: fedjax.create_model_from_haiku
.. autofunction:: fedjax.create_model_from_stax
.. autofunction:: fedjax.evaluate_model

----

.. autoclass:: fedjax.FederatedAlgorithm

----

.. autofunction:: fedjax.for_each_client
.. autofunction:: fedjax.for_each_client_backend
.. autofunction:: fedjax.set_for_each_client_backend
.. autoclass:: fedjax.ForEachClientBackend
	:special-members: __call__
