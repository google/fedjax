# FedJAX: Federated learning simulation with JAX

[![Build and minimal test](https://github.com/google/fedjax/actions/workflows/build_and_minimal_test.yml/badge.svg)](https://github.com/google/fedjax/actions/workflows/build_and_minimal_test.yml)
[![Documentation Status](https://readthedocs.org/projects/fedjax/badge/?version=latest)](https://fedjax.readthedocs.io/en/latest/?badge=latest)
![PyPI version](https://img.shields.io/pypi/v/fedjax)

[**Documentation**](https://fedjax.readthedocs.io/) |
[**Paper**](https://arxiv.org/abs/2108.02117)

NOTE: FedJAX is not an officially supported Google product. FedJAX is still in
the early stages and the API will likely continue to change.

## What is FedJAX?

FedJAX is a [JAX]-based open source library for
[Federated Learning](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)
simulations that emphasizes ease-of-use in research. With its simple primitives
for implementing federated learning algorithms, prepackaged datasets, models and
algorithms, and fast simulation speed, FedJAX aims to make developing and
evaluating federated algorithms faster and easier for researchers. FedJAX works
on accelerators (GPU and TPU) without much additional effort. Additional details
and benchmarks can be found in our [paper](https://arxiv.org/abs/2108.02117).

## Quickstart


The following tutorial notebooks provide an introduction to FedJAX:

*   [Federated datasets](https://fedjax.readthedocs.io/en/latest/notebooks/dataset_tutorial.html)
*   [Working with models in FedJAX](https://fedjax.readthedocs.io/en/latest/notebooks/model_tutorial.html)
*   [Federated learning algorithms](https://fedjax.readthedocs.io/en/latest/notebooks/algorithms_tutorial.html)

You can also take a look at some of our examples:

*   [Federated Averaging](examples/fed_avg.py)
*   [Full EMNIST example](examples/emnist_fed_avg.py)

Below, we walk through a simple example of
[federated averaging](https://arxiv.org/abs/1602.05629) for linear regression
implemented in FedJAX. The first steps are to set up the experiment by loading
the federated dataset, initializing the model parameters, and defining the loss
and gradient functions. The federated dataset can be thought of as a simple
mapping from client identifiers to each client's local dataset.

```python
import jax
import jax.numpy as jnp
import fedjax

# {'client_id': client_dataset}.
federated_data = fedjax.FederatedData()
# Initialize model parameters.
server_params = jnp.array(0.5)
# Mean squared error.
mse_loss = lambda params, batch: jnp.mean(
        (jnp.dot(batch['x'], params) - batch['y'])**2)
# jax.jit for XLA and jax.grad for autograd.
grad_fn = jax.jit(jax.grad(mse_loss))
```

Next, we use
[`fedjax.for_each_client`](https://fedjax.readthedocs.io/en/latest/fedjax.html#fedjax.for_each_client)
to coordinate the training that occurs across multiple clients. For federated
averaging, `client_init` initializes the client model using the server model,
`client_step` completes one step of local mini-batch SGD, and `client_final`
returns the difference between the initial server model and the trained client
model. By using `fedjax.for_each_client`, this work will run on any available
accelerators and possibly in parallel because it is backed by `jax.jit` and
`jax.pmap`. However, while this is already straightforward to write, the same
could also be written out as a basic for loop over clients if desired.

```python
# For loop over clients with client learning rate 0.1.
for_each_client = fedjax.for_each_client(
  client_init=lambda server_params, _: server_params,
  client_step=(
    lambda params, batch: params - grad_fn(params, batch) * 0.1),
  client_final=lambda server_params, params: server_params - params)
```

Finally, we run federated averaging for `100` training rounds by sampling
clients from the federated dataset, training across these clients using the
`fedjax.for_each_client`, and aggregating the client updates using weighted
averaging to update the server model.

```python
# 100 rounds of federated training.
for _ in range(100):
  clients = federated_data.clients()
  client_updates = []
  client_weights = []
  for client_id, update in for_each_client(server_params, clients):
    client_updates.append(update)
    client_weights.append(federated_data.client_size(client_id))
  # Weighted average of client updates.
  server_update = (
    jnp.sum(client_updates * client_weights) /
    jnp.sum(client_weights))
  # Server learning rate of 0.01.
  server_params = server_params - server_update * 0.01
```

## Installation

You will need a moderately recent version of Python. Please check
[the PyPI page](https://pypi.org/project/fedjax/) for the up to date version
requirement.

First, install JAX. For a CPU-only version:

```
pip install --upgrade pip
pip install --upgrade jax jaxlib  # CPU-only version
```

For other devices (e.g. GPU), follow
[these instructions](https://github.com/google/jax#installation).

Then, install FedJAX from PyPI:

```
pip install fedjax
```

Or, to upgrade to the latest version of FedJAX:

```
pip install --upgrade git+https://github.com/google/fedjax.git
```

## Citing FedJAX

To cite this repository:

```
@software{fedjax2020github,
  author = {Jae Hun Ro and Ananda Theertha Suresh and Ke Wu},
  title = {{F}ed{JAX}: Federated learning simulation with {JAX}},
  url = {http://github.com/google/fedjax},
  version = {0.0.6},
  year = {2020},
}
```

In the above bibtex entry, the version number is intended to be that from
[fedjax/version.py](fedjax/version.py), and the
year corresponds to the project's open-source release. There is also an
associated [paper](https://arxiv.org/abs/2108.02117).

## Useful pointers

*   https://jax.readthedocs.io/en/latest/index.html
*   https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html
*   https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html
*   https://dm-haiku.readthedocs.io/en/latest/

[JAX]: https://github.com/google/jax
[Haiku]: https://github.com/deepmind/dm-haiku
[Stax]: https://github.com/google/jax/blob/master/jax/experimental/stax.py
[Optax]: https://github.com/deepmind/optax
