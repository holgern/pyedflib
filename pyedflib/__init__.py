# flake8: noqa

# Copyright (c) 2015 - 2017 Holger Nahrstaedt
# Copyright (c) 2016-2017 The pyedflib Developers
#                         <https://github.com/holgern/pyedflib>
# See LICENSE for license details.

from __future__ import division, print_function, absolute_import

from ._extensions._pyedflib import *
from .edfwriter import *
from .edfreader import *

from . import data

from pyedflib.version import version as __version__
from numpy.testing import Tester

__all__ = [s for s in dir() if not s.startswith('_')]
try:
    # In Python 2.x the name of the tempvar leaks out of the list
    # comprehension.  Delete it to not make it show up in the main namespace.
    del s
except NameError:
    pass


test = Tester().test
