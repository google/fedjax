fedjax.training
===============

.. automodule:: fedjax.training

Federated experiment
--------------------

.. autofunction:: fedjax.training.run_federated_experiment

.. autoclass:: fedjax.training.FederatedExperimentConfig

.. autoclass:: fedjax.training.EvaluationFn
	:special-members: __call__

.. autoclass:: fedjax.training.ModelFullEvaluationFn
	:special-members: __init__, __call__
	:show-inheritance:

.. autoclass:: fedjax.training.ModelSampleClientsEvaluationFn
	:special-members: __init__, __call__
	:show-inheritance:

.. autoclass:: fedjax.training.TrainClientsEvaluationFn
	:special-members: __call__

.. autoclass:: fedjax.training.ModelTrainClientsEvaluationFn
	:special-members: __init__, __call__
	:show-inheritance:

.. autofunction:: fedjax.training.set_tf_cpu_only

.. autofunction:: fedjax.training.load_latest_checkpoint

.. autofunction:: fedjax.training.save_checkpoint

.. autoclass:: fedjax.training.Logger
	:members:
	:special-members: __init__

Tasks
-----

.. automodule:: fedjax.training.tasks

.. autoattribute:: fedjax.training.ALL_TASKS

.. autofunction:: fedjax.training.get_task

Structured flags
----------------

.. automodule:: fedjax.training.structured_flags
	:members: