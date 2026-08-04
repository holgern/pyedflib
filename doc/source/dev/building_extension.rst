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
    pip install --group docs .      # Install with documentation dependencies
    pip install --group dev .       # Install with all development dependencies

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

Version management
------------------

PyEDFlib uses setuptools-scm for automatic version management. The version is
derived from git tags:

- Tagged releases (e.g., ``v0.1.43``) produce version ``0.1.43``
- Development versions include git commit hash and date (e.g., ``0.1.43.dev35+gb22e717.d20260804``)
- No manual version updates are required in setup.py or any other files

The version can be accessed programmatically via ``pyedflib.__version__``.

