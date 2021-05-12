# How to build the docs

1. Install the requirements `pip install -r docs/requirements.txt`
2. Run `sphinx-autobuild -b html --watch . --open-browser docs docs/_build/html`

`sphinx-autobuild` will watch for file changes and auto build the HTML for you,
so all you'll have to do is refresh the page. If you don't want to use the auto
builder, you can just use `sphinx-build -b html docs docs/_build/html` and then navigate
to `docs/_build/html/index.html` in your browser.

# How to write code documentation

Our documentation it is written in ReStructuredText for Sphinx.
This is a meta-language that is compiled into online documentation.
For more details see
[Sphinx's documentation](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html).

We also rely heavily on [`sphinx.ext.autodoc`](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html) to convert docstrings in source to rst for Sphinx,
so it would be best to be familiar with its [directives](https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html).

As a result, our docstrings adhere to a specific syntax that has to be kept in mind.
Below we provide some guidelines.

## How to use "code font"

When writing code font in a docstring, please use double backticks.

```bash
# This returns a ``str`` object.
```

## How to create cross-references/links

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

## How to write math

We're using [`sphinx.ext.mathjax`](https://www.sphinx-doc.org/en/master/usage/extensions/math.html#module-sphinx.ext.mathjax).
Then we use the math [directive](https://www.sphinx-doc.org/en/3.x/usage/restructuredtext/directives.html#directive-math) and [role](https://www.sphinx-doc.org/en/3.x/usage/restructuredtext/roles.html#role-math) to either inline or block notation.


```bash
# Blocked notation
# .. math::
#		x + y

# Inline notation :math:`x + y`
```

## Examples

Take a look at [`docs/fedjax.experimental.metrics.rst`](fedjax.experimental.metrics.rst) to get a good idea of what to do.
