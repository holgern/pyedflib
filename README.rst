pyEDFlib
========

.. contents::

.. image:: https://codecov.io/gh/holgern/pyedflib/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/holgern/pyedflib
   :alt: Test Coverage

.. image:: https://readthedocs.org/projects/pyedflib/badge/?version=latest
   :target: https://pyedflib.readthedocs.io/en/latest/?badge=latest
   :alt: Docs Build

.. image:: https://badge.fury.io/py/pyEDFlib.svg
   :target: https://badge.fury.io/py/pyEDFlib
   :alt: PyPI Version

.. image:: https://img.shields.io/conda/vn/conda-forge/pyedflib.svg
   :target: https://anaconda.org/conda-forge/pyedflib
   :alt: Conda Version

.. image:: https://anaconda.org/conda-forge/pyedflib/badges/downloads.svg
   :target: https://anaconda.org/conda-forge/pyedflib
   :alt: Conda Downloads

What is pyEDFlib
----------------
pyEDFlib is a python library to read/write EDF+/BDF+ files based on EDFlib.

EDF means `European Data Format`_ and was firstly published `Kemp1992`_.
In 2003, an improved version of the file protocol  named EDF+ has been published and can be found at `Kemp2003`_.

The EDF/EDF+ format saves all data with 16 Bit. The company `BioSemi`_
introduced a version which saves all data with 24 Bit.

The definition of the EDF/EDF+/BDF/BDF+ format can be found under `edfplus.info`_.

This Python toolbox is a fork of the `toolbox from Christopher Lee-Messer`_
and uses the `EDFlib`_ from Teunis van Beelen.
The EDFlib is able to read and write EDF/EDF+/BDF/BDF+ files.

Documentation
-------------

Documentation is available online at https://pyedflib.readthedocs.io.

Installation
------------

pyEDFlib can be used with `Python`_ >=3.7. It depends on the `Numpy`_ package.
To use the newest source code from git, you have to download the source code.
You need a C compiler and a recent version of `Cython`_. Go then to the source directory and type::

    python setup.py build
    python setup.py install

There are binary wheels which can be installed by (use pip3 when available)::

    pip install pyEDFlib

Users of the Anaconda_ Python distribution can directly obtain pre-built
Windows, Intel Linux or macOS / OSX binaries from the conda-forge channel.
This can be done via::

    conda install -c conda-forge pyedflib


The most recent *development* version can be found on GitHub at
https://github.com/holgern/pyedflib.

The latest release, including source and binary packages for Linux,
macOS and Windows, is available for download from the `Python Package Index`_.
You can find source releases at the `Releases Page`_.


Highlevel interface
-------------------

pyEDFlib includes an highlevel interface for easy access to read and write edf files.
Additionally functionality as anonymizing, dropping or renaming channels can be found there.

.. code-block:: Python

    from pyedflib import highlevel

    # write an edf file
    signals = np.random.rand(5, 256*300)*200 # 5 minutes of random signal
    channel_names = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5']
    signal_headers = highlevel.make_signal_headers(channel_names, sample_frequency=256)
    header = highlevel.make_header(patientname='patient_x', gender='Female')
    highlevel.write_edf('edf_file.edf', signals, signal_headers, header)

    # read an edf file
    signals, signal_headers, header = highlevel.read_edf('edf_file.edf', ch_names=['ch1', 'ch2'])
    print(signal_headers[0]['sample_frequency']) # prints 256

    # drop a channel from the file or anonymize edf
    highlevel.drop_channels('edf_file.edf', to_drop=['ch2', 'ch4'])
    highlevel.anonymize_edf('edf_file.edf', new_file='anonymized.edf'
                             to_remove=['patientname', 'birthdate'],
                             new_values=['anonymized', ''])
    # check if the two files have the same content
    highlevel.compare_edf('edf_file.edf', 'anonymized.edf')
    # change polarity of certain channels
    highlevel.change_polarity('file.edf', channels=[1,3])
    # rename channels within a file
    highlevel.rename_channels('file.edf', mapping={'C3-M1':'C3'})


License
-------

pyEDFlib is a free Open Source software released under the BSD 2-clause license.


Releases can be cited via Zenodo.

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5678481.svg
   :target: https://doi.org/10.5281/zenodo.5678481

.. _Cython: https://cython.org/
.. _Anaconda: https://www.anaconda.com/
.. _GitHub: https://github.com/holgern/pyedflib
.. _GitHub Issues: https://github.com/holgern/pyedflib/issues
.. _Numpy: https://numpy.org/
.. _Python: https://www.python.org/
.. _Python Package Index: https://pypi.org/project/pyEDFlib/
.. _Releases Page: https://github.com/holgern/pyedflib/releases
.. _edfplus.info: https://www.edfplus.info/
.. _European Data Format: https://www.edfplus.info/
.. _Kemp2003: https://www.ncbi.nlm.nih.gov/pubmed/12948806?dopt=Citation
.. _Kemp1992: https://www.ncbi.nlm.nih.gov/pubmed/1374708?dopt=Abstract
.. _BioSemi: https://www.biosemi.com/faq/file_format.htm
.. _toolbox from Christopher Lee-Messer: https://github.com/cleemesser/python-edf
.. _EDFlib: https://www.teuniz.net/edflib/
