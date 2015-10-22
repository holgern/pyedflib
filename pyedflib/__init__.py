from __future__ import division, print_function, absolute_import

from ._pyedflib import *
from .edfwriter import *
from .edfreader import *

from pyedflib.version import version as __version__

from numpy.testing import Tester

__all__ = [s for s in dir() if not s.startswith('_')]
test = Tester().test
