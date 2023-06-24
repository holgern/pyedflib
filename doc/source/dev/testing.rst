.. _dev-testing:

Testing
=======

We are currently using AppVeyor and CircleCI for continuous integration.

If you are submitting a patch or pull request please make sure it
does not break the build.


Running tests locally
---------------------

Tests are implemented with `nose`_, so use one of:

    $ nosetests pyedflib

    >>> pyedflib.test()  # doctest: +SKIP
    
Note doctests require `Matplotlib`_ in addition to the usual dependencies.


.. _nose: https://nose.readthedocs.io/
.. _Matplotlib: https://matplotlib.org/
