.. _dev-building-on-windows:


Preparing Windows build environment
===================================

To start developing pyedflib code on Windows you will have to install
a C compiler and prepare the build environment.

Installing Microsoft Visual C++ Compiler for Python 2.7
-------------------------------------------------------

Downloading  Microsoft Visual C++ Compiler for Python 2.7 from https://www.microsoft.com/en-us/download/details.aspx?id=44266.


After installing the Compiler and before compiling the extension you have
to configure some environment variables.

For  build execute the ``util/setenv_win.bat`` script in the cmd
window:

  .. sourcecode:: bat

    rem Configure the environment for builds.
    rem Convince setup.py to use the SDK tools.
    set MSSdk=1
    set DISTUTILS_USE_SDK=1



Next steps
----------

After completing these steps continue with
:ref:`Installing build dependencies <dev-installing-build-dependencies>`.


.. _Python: https://www.python.org/
.. _numpy: https://numpy.org/
.. _Cython: https://cython.org/
.. _Sphinx: https://www.sphinx-doc.org/
.. _Microsoft Visual C++ Compiler for Python 2.7: https://www.microsoft.com/en-us/download/details.aspx?id=44266
