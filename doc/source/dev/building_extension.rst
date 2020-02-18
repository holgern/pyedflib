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

    python setup.py build
    python setup.py install

To verify the installation run the following command::

    python setup.py test

To build docs::

    cd doc
    make html
	
Installing from source code in Windows
--------------------------------------

Go to https://github.com/holgern/pyedflib GitHub project page, fork and clone the
repository or use the upstream repository to get the source code::

    git clone https://github.com/holgern/pyedflib.git pyedflib

Install Microsoft Visual C++ Compiler for Python 2.7 from https://www.microsoft.com/en-us/download/details.aspx?id=44266

Activate your Python virtual environment, go to the cloned source directory
and type the following commands to build and install the package::

	util\setenv_win.bat
    python setup.py build_ext --inplace
    python setup.py install --user

To verify the installation run the following command::

    python runtests.py

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

