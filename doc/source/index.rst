PyEDFlib -EDF/BDF Toolbox in Python
=================================================

PyEDFlib is a free Open Source wavelet toolbox for reading / writing EDF/BDF files. 


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
-------------

pyEDFlib is a python library to read/write EDF+/BDF+ files based on EDFlib.

EDF means [European Data Format](http://www.edfplus.info/) and was firstly published [1992](http://www.sciencedirect.com/science/article/pii/0013469492900097). 
In 2003, an improved version of the file protokoll named EDF+ has been published and can be found [here](http://www.sciencedirect.com/science/article/pii/0013469492900097).

The EDF/EDF+ format saves all data with 16 Bit. A version which saves all data with 24 Bit,
was introduces by the compony [BioSemi](http://www.biosemi.com/faq/file_format.htm).

The definition of the EDF/EDF+/BDF/BDF+ format can be found under [edfplus.info](http://www.edfplus.info/).

This python toolbox is a fork of the [toolbox from Christopher Lee-Messer](https://bitbucket.org/cleemesser/python-edf/)
and uses the [EDFlib](http://www.teuniz.net/edflib/) from Teunis van Beelen.
The EDFlib is able to read and write EDF/EDF+/BDF/BDF+ files.


Requirements
------------

 It requires:

 - Python_ 2.7 or >=3.3
 - Numpy_ >= 1.6.2

Download
--------

The most recent *development* version can be found on GitHub at
https://github.com/holgern/pyedflib.

Latest release, including source and binary package for Windows, is available
for download from the `Python Package Index`_ or on the `Releases Page`_.

License
--------

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
