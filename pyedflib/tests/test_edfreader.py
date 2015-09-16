# -*- coding: utf-8 -*-
# Copyright (c) 2015 Holger Nahrstaedt
from __future__ import division, print_function, absolute_import

import os
import numpy as np
from numpy.testing import (assert_raises, run_module_suite,
                           assert_equal, assert_allclose)

import pyedflib

data_dir = os.path.join(os.path.dirname(__file__), 'data')
edf_data_file = os.path.join(data_dir, 'test_generator.edf')



def test_Edfino():
    f = pyedflib.edfreader.Edfinfo(edf_data_file)
    assert_equal(f.file_name,edf_data_file)
    
if __name__ == '__main__':
    run_module_suite()