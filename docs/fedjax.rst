fedjax core
===========

.. automodule:: fedjax

Subpackages
-----------

.. toctree::
  :maxdepth: 1

  fedjax.metrics
  fedjax.optimizers
  fedjax.tree_util

Federated algorithm
-------------------

.. autosummary::
  :nosignatures:

  fedjax.FederatedAlgorithm

Federated data
--------------

.. autosummary::
  :nosignatures:

  fedjax.FederatedData
  fedjax.SubsetFederatedData
  fedjax.SQLiteFederatedData
  fedjax.InMemoryFederatedData
  fedjax.FederatedDataBuilder
  fedjax.SQLiteFederatedDataBuilder
  fedjax.ClientPreprocessor
  fedjax.shuffle_repeat_batch_federated_data
  fedjax.padded_batch_federated_data

Client dataset
--------------

.. autosummary::
  :nosignatures:

  fedjax.ClientDataset
  fedjax.BatchPreprocessor
  fedjax.buffered_shuffle_batch_client_datasets
  fedjax.padded_batch_client_datasets

For each client
---------------

.. autosummary::
  :nosignatures:

  fedjax.for_each_client
  fedjax.for_each_client_backend
  fedjax.set_for_each_client_backend

Model
-----

.. autosummary::
  :nosignatures:

  fedjax.Model
  fedjax.create_model_from_haiku
  fedjax.create_model_from_stax
  fedjax.evaluate_model
  fedjax.model_grad
  fedjax.model_per_example_loss
  fedjax.evaluate_average_loss
  fedjax.ModelEvaluator
  fedjax.AverageLossEvaluator
  fedjax.grad

-------------------

.. autoclass:: fedjax.FederatedAlgorithm

--------------

.. autoclass:: fedjax.FederatedData
	:members:

.. autoclass:: fedjax.SubsetFederatedData
  :special-members: __init__
  :show-inheritance:

.. autoclass:: fedjax.SQLiteFederatedData
  :members: new
  :special-members: __init__
  :show-inheritance:

.. autoclass:: fedjax.InMemoryFederatedData
  :special-members: __init__
  :show-inheritance:

.. autoclass:: fedjax.FederatedDataBuilder
  :members:

.. autoclass:: fedjax.SQLiteFederatedDataBuilder
  :special-members: __init__
  :show-inheritance:

.. autoclass:: fedjax.ClientPreprocessor
  :members:
  :special-members: __init__, __call__

.. autofunction:: fedjax.shuffle_repeat_batch_federated_data

.. autofunction:: fedjax.padded_batch_federated_data

.. autoclass:: fedjax.RepeatableIterator

--------------

.. automodule:: fedjax.core.client_datasets

.. autoclass:: fedjax.ClientDataset
  :members:
  :special-members: __init__, __len__, __getitem__

.. autoclass:: fedjax.BatchPreprocessor
  :members:
  :special-members: __init__

.. autofunction:: fedjax.buffered_shuffle_batch_client_datasets

.. autofunction:: fedjax.padded_batch_client_datasets

---------------

.. autofunction:: fedjax.for_each_client

.. autofunction:: fedjax.for_each_client_backend

.. autofunction:: fedjax.set_for_each_client_backend

-----

.. autoclass:: fedjax.Model

.. autofunction:: fedjax.create_model_from_haiku

.. autofunction:: fedjax.create_model_from_stax

.. autofunction:: fedjax.evaluate_model

.. autofunction:: fedjax.model_grad

.. autofunction:: fedjax.model_per_example_loss

.. autofunction:: fedjax.evaluate_average_loss

.. autoclass:: fedjax.ModelEvaluator
  :members:
  :special-members: __init__

.. autoclass:: fedjax.AverageLossEvaluator
  :members:
  :special-members: __init__

.. autofunction:: fedjax.grad