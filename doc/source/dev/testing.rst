.. _dev-testing:

Testing
=======

We are currently using GitHub Actions for continuous integration.

If you are submitting a patch or pull request please make sure it
does not break the build.


Running tests locally
---------------------

Tests are implemented with `pytest`, so use one of:

    $ pytest

    >>> pyedflib.test()  # doctest: +SKIP

Note doctests require `Matplotlib`_ in addition to the usual dependencies.


Pre-commit hooks
-----------------

This project uses `pre-commit`_ to run code quality checks before commits.
The hooks include:

- Code formatting and linting with `ruff`_
- Various checks from `pre-commit-hooks`_ (trailing whitespace, end-of-file fixer, etc.)

To use pre-commit hooks, first install pre-commit::

    $ pip install pre-commit

Then install the git hooks::

    $ pre-commit install

After this, the hooks will run automatically on ``git commit``.
To run the hooks manually on all files::

    $ pre-commit run -a

The configuration is defined in ``.pre-commit-config.yaml`` at the root of the repository.


.. _Matplotlib: https://matplotlib.org/
.. _pre-commit: https://pre-commit.com/
.. _ruff: https://astral-sh.github.io/ruff/
.. _pre-commit-hooks: https://github.com/pre-commit/pre-commit-hooks
