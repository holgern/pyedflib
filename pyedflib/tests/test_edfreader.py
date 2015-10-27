# -*- coding: utf-8 -*-
# Copyright (c) 2015 Holger Nahrstaedt
from __future__ import division, print_function, absolute_import

import os
import numpy as np
from datetime import datetime, date
# from numpy.testing import (assert_raises, run_module_suite,
#                            assert_equal, assert_allclose, assert_almost_equal)
import sys
import unittest
import pyedflib


class TestEdfReader(unittest.TestCase):
    def setUp(self):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.edf_data_file = os.path.join(data_dir, 'test_generator.edf')

    def test_EdfReader(self):
        f = pyedflib.EdfReader(self.edf_data_file)
        ann_index, ann_duration, ann_text = f.readAnnotations()
        np.testing.assert_almost_equal(ann_index[0], 0)
        np.testing.assert_almost_equal(ann_index[1], 600)

        np.testing.assert_equal(f.signals_in_file, 11)
        np.testing.assert_equal(f.datarecords_in_file, 600)

        for i in np.arange(11):
            np.testing.assert_almost_equal(f.getSampleFrequencies()[i], 200)
            np.testing.assert_equal(f.getNSamples()[i], 120000)

        f._close()
        del f

    def test_EdfReader_headerInfos(self):
        f = pyedflib.EdfReader(self.edf_data_file)

        datetimeSoll = datetime(2011,4,4,12,57,2)
        np.testing.assert_equal(f.getStartdatetime(),datetimeSoll)
        np.testing.assert_equal(f.getPatientCode(), b'abcxyz99')
        np.testing.assert_equal(f.getPatientName(), b'Hans Muller')
        np.testing.assert_equal(f.getGender(), b'Male')
        np.testing.assert_equal(f.getBirthdate(), b'30 jun 1969')
        np.testing.assert_equal(f.getPatientAdditional(), b'patient')
        np.testing.assert_equal(f.getAdmincode(), b'Dr. X')
        np.testing.assert_equal(f.getTechnician(), b'Mr. Spotty')
        np.testing.assert_equal(f.getRecordingAdditional(), b'unit test file')
        np.testing.assert_equal(f.getFileDuration(), 600)
        fileHeader = f.getHeader()
        np.testing.assert_equal(fileHeader["patientname"], b'Hans Muller')
        f._close()
        del f

    def test_EdfReader_signalInfos(self):
        f = pyedflib.EdfReader(self.edf_data_file)
        np.testing.assert_equal(f.getSignalLabels()[0], b'squarewave')
        np.testing.assert_equal(f.getLabel(0), b'squarewave')
        np.testing.assert_equal(f.getPhysicalDimension(0), b'uV')
        np.testing.assert_equal(f.getPrefilter(0), b'pre1')
        np.testing.assert_equal(f.getTransducer(0), b'trans1')
        np.testing.assert_equal(f.getSampleFrequency(0), 200)
        np.testing.assert_equal(f.getSampleFrequencies()[0], 200)

        np.testing.assert_equal(f.getSignalLabels()[1], b'ramp')
        np.testing.assert_equal(f.getSignalLabels()[2], b'pulse')
        np.testing.assert_equal(f.getSignalLabels()[3], b'noise')
        np.testing.assert_equal(f.getSignalLabels()[4], b'sine 1 Hz')
        np.testing.assert_equal(f.getSignalLabels()[5], b'sine 8 Hz')
        f._close()
        del f

if __name__ == '__main__':
    # run_module_suite(argv=sys.argv)
    unittest.main()
