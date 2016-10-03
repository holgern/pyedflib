from __future__ import division, print_function, absolute_import

from ._extensions._pyedflib import *
from .edfwriter import *
from .edfreader import *
from pyedflib.version import version as __version__
from numpy.testing import Tester

from . import data

__all__ = [s for s in dir() if not s.startswith('_')]
try:
    # In Python 2.x the name of the tempvar leaks out of the list
    # comprehension.  Delete it to not make it show up in the main namespace.
    del s
except NameError:
    pass


test = Tester().test
