# Frequently Asked Questions

## How can I contribute?

See the [README](./README.md) and
[guidelines for contributors](./CONTRIBUTING.md).

## Can FedJAX be used in a production setting, e.g., on mobile phones?

Unfortunately, no. FedJAX is a lightweight Python- and JAX-based simulation
library that focuses on ease of use and rapid prototyping of federated learning
algorithms for research purposes.

## What is the relationship between FedJAX and TensorFlow Federated?

[TensorFlow Federated (TFF)](https://www.tensorflow.org/federated) is a
full-fledged framework for federated learning that is designed to facilitate
composing different algorithms and features, and to enable porting code across
different simulation and deployment scenarios. TFF provides a scalable runtime
and supports several privacy, compression algorithms, and optimizers, as
outlined in https://www.tensorflow.org/federated/tff_for_research. In contrast,
FedJAX is a lightweight Python- and JAX-based simulation library that focuses on
ease of use and rapid prototyping of federated learning algorithms for research
purposes. TensorFlow Federated and FedJAX are developed as separate projects,
without expectation of code portability.
