#!/bin/bash
#
# Script for running tests in OSS development.
#
# Examples:
#
# * Running all tests under fedjax/ (default):
#
#     ./run_tests.sh
#
# * Running tests in directory fedjax/core:
#
#     ./run_tests.sh fedjax/core
#
# * Running tests in fedjax/fedjax_test.py and directory fedjax/core:
#
#     ./run_tests.sh fedjax/fedjax_test.py fedjax/core

set -e
set -o pipefail

if [[ "$#" -eq 0 ]]; then
  TO_RUN=(fedjax)
else
  TO_RUN="$@"
fi

# Install build/test dependencies.
pip install -e .
pip install -r requirements-test.txt

# Run tests in serial.
#
# We use 'python -I' to prevent the script directory from being included in
# sys.path.
#
# TODO(wuke): Improve readability of test failures.
#
# TODO(wuke): Run tests in parallel. We need to deal with race conditions caused
# by multiple tests accessing temporary files with the same name (e.g.
# "test_sqlite_federated_data.sqlite" in both
# fedjax/core/sqlite_federated_data_test.py and
# fedjax/core/federated_data_test.py
find "${TO_RUN[@]}" -name '*_test.py' -print0 | \
  CUDA_VISIBLE_DEVICES=-1 xargs -0 -n1 -- sh -c 'python -I $0 2>&1 || exit 255'
