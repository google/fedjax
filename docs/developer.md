# Building from source

First, clone the FedJAX source code:

```sh
git clone https://github.com/google/fedjax
cd fedjax
```

Then install the `fedjax` Python package:

```sh
pip install -e .
```

To upgrade to the latest version from GitHub, inside of the repository root,
just run

```sh
git pull
```

You shouldn't have to reinstall `fedjax` because `pip install -e` sets up
symbolic links from site-packages into the repository.

# Running the tests

## Running specific tests

You can run a specific set of tests using
[pytest](https://docs.pytest.org/en/latest/usage.html#specifying-tests-selecting-tests)'s
built-in selection mechanisms, or alternatively, you can run a specific test
file directly to see more detailed information about the cases being run.

We recommend using `pytest-xdist`, which can run tests in parallel. First,
install `pytest-xdist` and other test-only dependencies:

```sh
pip install -r requirements-test.txt
```

Then, from the repository root directory, run:

```sh
# Run all tests in algorithms
pytest -n auto fedjax/algorithms

# Run only fedjax/core/metrics_test.py
pytest -n auto fedjax/core/metrics_test.py
```

The `-n auto` tells pytest to use as many processes as your computer has CPU
cores. For more details, see the
[pytest-xdist docs](https://github.com/pytest-dev/pytest-xdist#speed-up-test-runs-by-sending-tests-to-multiple-cpus).

We also use a handful of custom configurations in the pytest.ini file. For more
details, see the
[pytest docs](https://docs.pytest.org/en/6.2.x/reference.html#ini-options-ref).

## Running all CI tests

Before creating a pull request, we recommend running all the FedJAX tests that
are used in the continuous integration:

```sh
pytest -n auto -q \
  -k "not SubsetFederatedDataTest and not SQLiteFederatedDataTest and not ForEachClientPmapTest and not DownloadsTest and not CheckpointTest and not LoggingTest" \
  fedjax --ignore=fedjax/legacy/
```

`-q` will reduce verbosity levels, `-k` selects/deselects specific tests, and
`--ignore=fedjax/legacy/` is used to skip the entire fedjax.legacy module. If
there are errors or failures, you can run those specific tests using the
commands in the previous section to see more focused details.
