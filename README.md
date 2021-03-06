# FedJAX: Federated learning simulation with JAX

NOTE: FedJAX is not an officially supported Google product. FedJAX is still in
the early stages and the API will likely continue to change.

## What is FedJAX?

FedJAX is a library for developing custom
[Federated Learning (FL)](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)
algorithms in [JAX]. FedJAX prioritizes ease-of-use and is intended to be useful
for anyone with knowledge of NumPy.

FedJAX is built around the common core components needed in the FL setting:

*   **Federated datasets**: Clients and a dataset for each client
*   **Models**: CNN, ResNet, etc.
*   **Optimizers**: SGD, Momentum, etc.
*   **Federated algorithms**: Client updates and server aggregation

For **Models** and **Optimizers**, FedJAX provides lightweight wrappers and
containers that can work with a variety of existing implementations (e.g. a
model wrapper that can support both [Haiku] and [Stax]). Similarly, for
**Federated datasets**, [TFF] provides a well established API for working with
federated datasets, and FedJAX just provides utilties for converting to NumPy
input acceptable to JAX.

However, what FL researchers will find most useful is the collection and
customizability of **Federated algorithms** provided out of box by FedJAX.

## Quickstart

The
[FedJAX Intro notebook](notebooks/fedjax_intro.ipynb)
provides an introduction into writing and running FedJAX experiments.

You can also take a look at some of our examples:

*   [Simple Federated Averaging](examples/simple_fed_avg.py)
*   [Full EMNIST example](examples/emnist_simple_fed_avg.py).

## Installation

You will need Python 3.6 or later and a working JAX installation. For a CPU-only
version:

```
pip install --upgrade pip
pip install --upgrade jax jaxlib  # CPU-only version
```

For other devices (e.g. GPU), follow
[these instructions](https://github.com/google/jax#installation).

Then, install fedjax from PyPi:

```
pip install fedjax
```

Or, to upgrade to the latest version of fedjax:

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
  version = {0.0.3},
  year = {2020},
}
```

In the above bibtex entry, the version number is intended to be that from
[fedjax/version.py](fedjax/version.py), and the
year corresponds to the project's open-source release.

## Useful pointers

*   https://jax.readthedocs.io/en/latest/index.html
*   https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html
*   https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html
*   https://dm-haiku.readthedocs.io/en/latest/
*   https://www.tensorflow.org/federated/api_docs/python/tff/simulation/ClientData

[JAX]: https://github.com/google/jax
[TFF]: https://www.tensorflow.org/federated
[Haiku]: https://github.com/deepmind/dm-haiku
[Stax]: https://github.com/google/jax/blob/master/jax/experimental/stax.py

