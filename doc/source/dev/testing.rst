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


.. _Matplotlib: https://matplotlib.org/
