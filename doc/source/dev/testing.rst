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


Running tests with Tox
----------------------

There's also a config file for running tests with `Tox`_ (``pip install tox``).
To for example run tests for Python 2.7 and Python 3.4 use::

  tox -e py27,py34

For more information see the `Tox`_ documentation.


.. _nose: http://nose.readthedocs.org/en/latest/ 
.. _Tox: http://tox.testrun.org/ 
.. _Matplotlib: http://matplotlib.org
