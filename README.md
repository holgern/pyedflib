pyEDFlib
========

.. contents::

What is pyEDFlib
----------------
pyEDFlib is a python library to read/write EDF+/BDF+ files based on EDFlib.

EDF means [European Data Format](http://www.edfplus.info/) and was firstly published [1992](http://www.sciencedirect.com/science/article/pii/0013469492900097). 
In 2003, an improved version of the file protokoll named EDF+ has been published and can be found [here](http://www.sciencedirect.com/science/article/pii/0013469492900097).

The EDF/EDF+ format saves all data with 16 Bit. A version which saves all data with 24 Bit,
was introduces by the compony [BioSemi](http://www.biosemi.com/faq/file_format.htm).

The definition of the EDF/EDF+/BDF/BDF+ format can be found under [edfplus.info](http://www.edfplus.info/).

This python toolbox is a fork of the [toolbox from Christopher Lee-Messer](https://bitbucket.org/cleemesser/python-edf/)
and uses the [EDFlib](http://www.teuniz.net/edflib/) from Teunis van Beelen.
The EDFlib is able to read and write EDF/EDF+/BDF/BDF+ files.

Documentation
-------------

Documentation is available online at http://pyedflib.readthedocs.org.

Installation
------------

pyEDFlib can be used with python 2.7.x or >=3.4. It depends on the numpy package.
To use the newest source code from git, you have to download the source code.
You need a C compiler and a recent version of Cython. Go then to the source directory and type:

    python setup.py build
    python setup.py install

There are binary wheels which can be installed by

    pip install pyEDFlib

License
-------

pyEDFlib is a free Open Source software released under the BSD 2-clause license.

Badges
------

[![codecov](https://codecov.io/gh/holgern/pyedflib/branch/master/graph/badge.svg)](https://codecov.io/gh/holgern/pyedflib)

[![Build Status](https://travis-ci.org/holgern/pyedflib.svg)](https://travis-ci.org/holgern/pyedflib)

[![Build status](https://ci.appveyor.com/api/projects/status/49wwigslgtj288q1?svg=true)](https://ci.appveyor.com/project/HolgerNahrstaedt/pyedflib)

[![Documentation Status](https://readthedocs.org/projects/pyedflib/badge/?version=latest)](http://pyedflib.readthedocs.org/en/latest/?badge=latest)