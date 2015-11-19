# pyEDFlib
pyEDFlib is a python library to read/write EDF+/BDF+ files based on EDFlib.

EDF means [European Data Format](http://www.edfplus.info/) and was firstly published [1992](http://www.sciencedirect.com/science/article/pii/0013469492900097). 
In 2003, an improved version of the file protokoll named EDF+ has been published and can be found [here](http://www.sciencedirect.com/science/article/pii/0013469492900097).

The EDF/EDF+ format saves all data with 16 Bit. A version which saves all data with 24 Bit,
was introduces by the compony [BioSemi](http://www.biosemi.com/faq/file_format.htm).

The definition of the EDF/EDF+/BDF/BDF+ format can be found under [edfplus.info](http://www.edfplus.info/).

This python toolbox is a fork of the [toolbox from Christopher Lee-Messer](https://bitbucket.org/cleemesser/python-edf/)
and uses the [EDFlib](http://www.teuniz.net/edflib/) from Teunis van Beelen.
The EDFlib is able to read and write EDF/EDF+/BDF/BDF+ files.

Authors
* Holger Nahrstaedt
* Teunis van Beelen (edflib.c, edflib.h, http://www.teuniz.net/edflib/)
* Chris Lee-Messer (https://bitbucket.org/cleemesser/python-edf)



[![Coverage Status](https://coveralls.io/repos/holgern/pyedflib/badge.svg?branch=master&service=github)](https://coveralls.io/github/holgern/pyedflib?branch=master)

[![Build Status](https://travis-ci.org/holgern/pyedflib.svg)](https://travis-ci.org/holgern/pyedflib)

[![Build status](https://ci.appveyor.com/api/projects/status/49wwigslgtj288q1?svg=true)](https://ci.appveyor.com/project/HolgerNahrstaedt/pyedflib)

[![Documentation Status](https://readthedocs.org/projects/pyedflib/badge/?version=latest)](http://pyedflib.readthedocs.org/en/latest/?badge=latest)