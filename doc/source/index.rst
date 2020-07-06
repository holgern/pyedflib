PyEDFlib -EDF/BDF Toolbox in Python
===================================

PyEDFlib is a free Open Source wavelet toolbox for reading / writing *EDF/EDF+/BDF* files.


  .. sourcecode:: python

    import pyedflib
    import numpy as np
    import os

    file_name = os.path.join(pyedflib.util.test_data_path(),
                             'test_generator.edf')
    f = pyedflib.EdfReader(file_name)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    sigbufs = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
        sigbufs[i, :] = f.readSignal(i)


Description
-----------

``PyEDFlib`` is a `Python`_ library to read/write *EDF/EDF+/BDF* files based on EDFlib.

*EDF* stands for `European Data Format <http://www.edfplus.info/>`_, a data format for EEG data, first `published in 1992 <https://doi.org/10.1016/0013-4694(92)90009-7>`_.
In 2003, an improved version of the file protocol named *EDF+* `has been published <https://doi.org/10.1016/S1388-2457(03)00123-8>`_.

The definition of the *EDF/EDF+* format can be found under `edfplus.info <https://www.edfplus.info/>`_.

The *EDF/EDF+* format saves all data with 16 Bit.
A version of the format which saves all data with 24 Bit, called *BDF*, was introduced by the company `BioSemi <https://www.biosemi.com/faq/file_format.htm>`_.

The ``PyEDFlib`` `Python`_ toolbox is a fork of the ``python-edf`` `toolbox from Christopher Lee-Messer <https://bitbucket.org/cleemesser/python-edf/>`_.
and uses the `EDFlib <http://www.teuniz.net/edflib/>`_ from Teunis van Beelen.

Requirements
------------

``PyEDFlib`` requires:

- Python_ >=3.5
- Numpy_ >= 1.9.1

Download
--------

The most recent *development* version can be found on GitHub at
https://github.com/holgern/pyedflib.

The latest release, including source and binary package for Windows, is available
for download from the `Python Package Index`_ or on the `Releases Page`_.

License
-------

This code is licensed under the same BSD-style license that Teunis released `edflib`_ under and with the same disclaimer.

Contents
--------

.. toctree::
   :maxdepth: 1

   ref/index
   dev/index
   resources
   contents

.. _Cython: http://cython.org/
.. _demo: https://github.com/holgern/pyedflib/tree/master/demo
.. _GitHub: https://github.com/holgern/pyedflib
.. _GitHub Issues: https://github.com/holgern/pyedflib/issues
.. _Numpy: http://www.numpy.org
.. _edflib: http://www.teuniz.net/edflib/
.. _Python: http://python.org/
.. _Python Package Index: http://pypi.python.org/pypi/pyedflib/
.. _Releases Page: https://github.com/holgern/pyedflib/releases
