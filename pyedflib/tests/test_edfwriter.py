# -*- coding: utf-8 -*-
# Copyright (c) 2015 Holger Nahrstaedt
from __future__ import division, print_function, absolute_import

import os
import numpy as np
# from numpy.testing import (assert_raises, run_module_suite,
#                            assert_equal, assert_allclose, assert_almost_equal)
import sys
import unittest
import pyedflib


class EdfWriterTest(unittest.TestCase):
    def setUp(self):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.bdf_data_file = os.path.join(data_dir, 'tmp_test_file.bdf')

    def test_EdfWriter(self):
        channel_info = {'label':'test_label', 'dimension':'mV', 'sample_rate':100,
                         'physical_max':1.0,'physical_min':-1.0,
                         'digital_max':8388607,'digital_min':-8388608,
                         'prefilter':'pre1','transducer':'trans1'}
        f = pyedflib.EdfWriter(self.bdf_data_file, 1,
                              file_type=pyedflib.FILETYPE_BDFPLUS)
        f.setChannelInfo(0,channel_info)
        f.setTechnician('tec1')
        data = np.ones(100) * 0.1
        f.writePhyisicalSamples(data)
        f.writePhyisicalSamples(data)
        f.close()
        del f

        f = pyedflib.EdfReader(self.bdf_data_file)
        np.testing.assert_equal(f.technician.rstrip(), b'tec1')

        np.testing.assert_equal(f.getLabel(0), b'test_label')
        np.testing.assert_equal(f.getPhysicalDimension(0), b'mV')
        np.testing.assert_equal(f.getPrefilter(0), b'pre1')
        np.testing.assert_equal(f.getTransducer(0), b'trans1')
        np.testing.assert_equal(f.getSampleFrequency(0), 100)
        f._close()
        del f

if __name__ == '__main__':
    # run_module_suite(argv=sys.argv)
    unittest.main()