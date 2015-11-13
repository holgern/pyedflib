PyEDFlib -EDF/BDF Toolbox in Python
=================================================

PyEDFlib is a free Open Source wavelet toolbox for reading / writing EDF/BDF files. 


  .. sourcecode:: python

    >>> import pyedflib
    >>> f = pyedflib.EdfReader("data/test_generator.edf")
    >>> n = f.signals_in_file
    >>> signal_labels = f.getSignalLabels(
    >>> sigbufs = np.zeros((n, f.getNSamples()[0]))
    >>>  for i in np.arange(n):
    >>>    sigbufs[i, :] = f.readSignal(i)


Main features
-------------

The main features of PyEDFlib are:

  * reading writing EDF / BDF files
  * PyEDFlib is based on the library EDFlib from Teunis van Beelen (http://www.teuniz.net/edflib/)

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



Contents
--------

.. toctree::
   :maxdepth: 1

   dev/index
   resources


.. _Cython: http://cython.org/
.. _demo: https://github.com/holgern/pyedflib/tree/master/demo
.. _GitHub: https://github.com/holgern/pyedflib
.. _GitHub Issues: https://github.com/holgern/pyedflib/issues
.. _Numpy: http://www.numpy.org
.. _original developer: http://www.teuniz.net/edflib/
.. _Python: http://python.org/
.. _Python Package Index: http://pypi.python.org/pypi/pyedflib/
.. _Releases Page: https://github.com/holgern/pyedflib/releases
