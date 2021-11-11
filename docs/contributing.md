# Contributing

Everyone can contribute to FedJAX, and we value everyone's contributions. There are several
ways to contribute, including:

- Improving or expanding FedJAX's [documentation](http://fedjax.readthedocs.io/)
- Contributing to FedJAX's [code-base](http://github.com/google/fedjax/)

The FedJAX project follows [Google's Open Source Community Guidelines](https://opensource.google/conduct/).

## Ways to contribute

We welcome pull requests, in particular for those issues marked with
[contributions welcome](https://github.com/google/fedjax/issues?q=is%3Aopen+is%3Aissue+label%3A%22contributions+welcome%22) or
[good first issue](https://github.com/google/fedjax/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22).

For other proposals, we ask that you first open a GitHub
[Issue](https://github.com/google/fedjax/issues/new/choose) or
[Discussion](https://github.com/google/fedjax/discussions)
to seek feedback on your planned contribution.

## Contributing code using pull requests

We do all of our development using git, so basic knowledge is assumed.

Follow these steps to contribute code:

1. Fork the FedJAX repository by clicking the **Fork** button on the
   [repository page](http://www.github.com/google/fedjax). This creates
   a copy of the FedJAX repository in your own account.

2. Install a support version Python listed in
   https://github.com/google/fedjax/blob/main/setup.py.

3. `pip` installing your fork from source. This allows you to modify the code
   and immediately test it out:

   ```bash
   git clone https://github.com/YOUR_USERNAME/fedjax
   cd fedjax
   pip install -r requirements-test.txt # Installs all testing requirements.
   pip install -e .  # Installs FedJAX from the current directory in editable mode.
   ```

4. Add the FedJAX repo as an upstream remote, so you can use it to sync your
   changes.

   ```bash
   git remote add upstream http://www.github.com/google/fedjax
   ```

5. Create a branch where you will develop from:

   ```bash
   git checkout -b name-of-change
   ```

   And implement your changes using your favorite editor. If you are adding
   a new algorithm, please add unit tests and an associated binary in the
   experiments folder. The binary should use the EMNIST dataset with a
   convolution neural model [example](https://github.com/google/fedjax/blob/main/examples/emnist_fed_avg.py)
   and have reasonable default hyperparameters that ideally reproduce results
   from a published paper. We strongly recommend using
   [`fedjax.for_each_client`](https://fedjax.readthedocs.io/en/latest/fedjax.html#fedjax.for_each_client)
   in your algorithm implementations for computational efficiency.

6. Make sure the tests pass by running the following command from the top of
   the repository:

  ```bash
  pytest -n auto -q \
    -k "not SubsetFederatedDataTest and not SQLiteFederatedDataTest and not ForEachClientPmapTest and not DownloadsTest and not CheckpointTest and not LoggingTest" \
    fedjax --ignore=fedjax/legacy/
  ```

  `-q` will reduce verbosity levels, `-k` selects/deselects specific tests, and
  `--ignore=fedjax/legacy/` is used to skip the entire fedjax.legacy module. If
  there are errors or failures, you can run those specific tests using the
  commands in the next section to see more focused details.

   ```bash
   pytest -n auto tests/
   ```

   If you know the specific test file that covers your
   changes, you can limit the tests to that; for example:

  ```bash
  # Run all tests in algorithms
  pytest -n auto fedjax/algorithms

  # Run only fedjax/core/metrics_test.py
  pytest -n auto fedjax/core/metrics_test.py
  ```

7. Once you are satisfied with your change, create a commit as follows (
   [how to write a commit message](https://chris.beams.io/posts/git-commit/)):

   ```bash
   git add file1.py file2.py ...
   git commit -m "Your commit message"
   ```

   Then sync your code with the main repo:

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

   Finally, push your commit on your development branch and create a remote
   branch in your fork that you can use to create a pull request from:

   ```bash
   git push --set-upstream origin name-of-change
   ```

8. Create a pull request from the FedJAX repository and send it for review.
   Check the {ref}`pr-checklist` for considerations when preparing your PR, and
   consult [GitHub Help](https://help.github.com/articles/about-pull-requests/)
   if you need more information on using pull requests.

(pr-checklist)=

## FedJAX pull request checklist

As you prepare a FedJAX pull request, here are a few things to keep in mind:

### Google contributor license agreement

Contributions to this project must be accompanied by a Google Contributor License
Agreement (CLA). You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again. If you're not certain whether you've signed a CLA, you can open your PR
and our friendly CI bot will check for you.

### Single-change commits and pull requests

A git commit ought to be a self-contained, single change with a descriptive
message. This helps with review and with identifying or reverting changes if
issues are uncovered later on.

**Pull requests typically comprise a single git commit.** (In some cases, for
instance for large refactors or internal rewrites, they may contain several.)
In preparing a pull request for review, you may need to squash together
multiple commits. We ask that you do this prior to sending the PR for review if
possible. The `git rebase -i` command might be useful to this end.

### Linting and Type-checking

Please follow the style guide and check code quality by pylint as stated here
https://google.github.io/styleguide/pyguide.html

### Full GitHub test suite

Your PR will automatically be run through a full test suite on GitHub CI, which
covers a range of Python versions, dependency versions, and configuration options.
It's normal for these tests to turn up failures that you didn't catch locally; to
fix the issues you can push new commits to your branch.

### Restricted test suite

Once your PR has been reviewed, a FedJAX maintainer will mark it as `Pull Ready`. This
will trigger a larger set of tests, including tests on GPU and TPU backends that are
not available via standard GitHub CI. Detailed results of these tests are not publicly
viweable, but the FedJAX mantainer assigned to your PR will communicate with you regarding
any failures these might uncover; it's not uncommon, for example, that numerical tests
need different tolerances on TPU than on CPU.
