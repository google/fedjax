# Building from source

First, clone the FedJAX source code:

```bash
git clone https://github.com/google/fedjax
cd fedjax
```

Then install the `fedjax` Python package:

```bash
pip install -e .
```

To upgrade to the latest version from GitHub, inside of the repository root,
just run

```bash
git pull
```

You shouldn't have to reinstall `fedjax` because `pip install -e` sets up
symbolic links from site-packages into the repository.

# Running the tests

We created a simple [`run_tests.sh`](https://www.google.com/url?sa=D&q=https%3A%2F%2Fgithub.com%2Fgoogle%2Ffedjax%2Fblob%2Fmain%2Frun_tests.sh)
script for running tests. See its comments for examples. Before creating a pull
request, we recommend running all the FedJAX tests (i.e. running `run_tests.sh`
with no arguments) to verify the correctness of a change.

# Updating the docs

Install the requirements

```bash
pip install -r docs/requirements.txt
```

Then run

```bash
sphinx-autobuild -b html --watch . --open-browser docs docs/_build/html
```

`sphinx-autobuild` will watch for file changes and auto build the HTML for you,
so all you'll have to do is refresh the page. If you don't want to use the auto
builder, you can just use:

```bash
sphinx-build -b html docs docs/_build/html
```

and then navigate to `docs/_build/html/index.html` in your browser.

## How to write code documentation

Our documentation it is written in ReStructuredText for Sphinx.
This is a meta-language that is compiled into online documentation.
For more details see
[Sphinx's documentation](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html).

We also rely heavily on [`sphinx.ext.autodoc`](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html) to convert docstrings in source to rst for Sphinx,
so it would be best to be familiar with its [directives](https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html).

As a result, our docstrings adhere to a specific syntax that has to be kept in mind.
Below we provide some guidelines.

### How to use "code font"

When writing code font in a docstring, please use double backticks.

```bash
# This returns a ``str`` object.
```

### How to create cross-references/links

It is possible to create cross-references to other classes, functions, and
methods. In the following, `obj_typ` is either `class`, `func`, or `meth`.

```bash
# First method:
# <obj_type>:`path_to_obj`

# Second method:
# :<obj_type>:`description <path_to_obj>`
```

You can use the second method if the `path_to_obj` is very long.

```bash
# Create: a reference to class fedjax.experimental.model.Model.
# :class:`fedjax.experimental.model.Model`

# Create a reference to local function my_func.
# :func:`my_func`

# Create a reference "Module.apply()" to method fedjax.experimental.model.Model.apply_for_train.
# :meth:`Model.apply_for_train <fedjax.experimental.model.Model.apply_for_train>`
```

To create a hyperlink, use the following syntax:
```bash
# Note the double underscore at the end:
# `Link to Google <http://www.google.com>`__
```

You can also cross reference `jax` documentation directly since we've added it via
[`sphinx.ext.intersphinx`](https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html) in `docs/conf.py`

```bash
# :func:`jax.jit`
# Links to https://jax.readthedocs.io/en/latest/jax.html#jax.jit
```

### How to write math

We're using [`sphinx.ext.mathjax`](https://www.sphinx-doc.org/en/master/usage/extensions/math.html#module-sphinx.ext.mathjax).
Then we use the math [directive](https://www.sphinx-doc.org/en/3.x/usage/restructuredtext/directives.html#directive-math) and [role](https://www.sphinx-doc.org/en/3.x/usage/restructuredtext/roles.html#role-math) to either inline or block notation.


```bash
# Blocked notation
# .. math::
#		x + y

# Inline notation :math:`x + y`
```

### Examples

Take a look at [`docs/fedjax.metrics.rst`](fedjax.metrics.rst) to get a good idea of what to do.

## Update notebooks

It is easiest to edit the notebooks in Jupyter or in Colab.
To edit notebooks in the Colab interface,
open <http://colab.research.google.com> and `Upload ipynb` from your local repo.
Update it as needed, `Run all cells` then `Download ipynb` to your local repo.
You may want to test that it executes properly, using `sphinx-build` as explained above.
We recommend making changes this way to avoid introducing format errors into the `.ipynb` files.

In the future, we may build and re-execute the notebooks as part of the
[Read the docs](https://fedjax.readthedocs.io/en/latest) build.
However, for now, we exclude all notebooks from the build due to long durations (downloading dataset files, expensive model training, etc.).
See `exclude_patterns` in [conf.py](https://github.com/google/fedjax/blob/main/docs/conf.py)
