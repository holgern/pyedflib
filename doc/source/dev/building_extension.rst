.. _dev-building-extension:

Building and installing PyEDFlib
================================

Installing from source code
---------------------------

Go to https://github.com/holgern/pyedflib GitHub project page, fork and clone the
repository or use the upstream repository to get the source code::

    git clone https://github.com/holgern/pyedflib.git pyedflib

Activate your Python virtual environment, go to the cloned source directory
and type the following commands to build and install the package::

    pip install .

To install with optional dependency groups::

    pip install --group test .      # Install with test dependencies
    pip install --group dev .       # Install with development dependencies
    pip install --group docs .      # Install with documentation dependencies

To verify the installation run the following command::

    python -m pytest

To build docs::

    cd doc
    make html
	
Installing from source code in Windows
--------------------------------------

Go to https://github.com/holgern/pyedflib GitHub project page, fork and clone the
repository or use the upstream repository to get the source code::

    git clone https://github.com/holgern/pyedflib.git pyedflib

Install Microsoft Visual C++ Compiler from https://visualstudio.microsoft.com/downloads/

Activate your Python virtual environment, go to the cloned source directory
and type the following commands to build and install the package::

    pip install -e .

To verify the installation run the following command::

    python -m pytest

To build docs::

    cd doc
    make html

Installing a development version
--------------------------------

You can also install directly from the source repository::

    pip install -e git+https://github.com/holgern/pyedflib.git#egg=pyedflib

or::

    pip install pyedflib==dev


Installing a regular release from PyPi
--------------------------------------

A regular release can be installed with pip or easy_install::

    pip install pyedflib

