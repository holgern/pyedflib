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


class EdfReaderTest(unittest.TestCase):
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
            np.testing.assert_almost_equal(f.getSignalFrequencies()[i], 200)
            np.testing.assert_equal(f.getNSamples()[i], 120000)

        f._close()
        del f

    def test_EdfReader_headerInfos(self):
        f = pyedflib.EdfReader(self.edf_data_file)
        np.testing.assert_equal(f.startdate_day, 4)
        np.testing.assert_equal(f.startdate_month, 4)
        np.testing.assert_equal(f.startdate_year, 2011)
        np.testing.assert_equal(f.starttime_hour, 12)
        np.testing.assert_equal(f.starttime_minute, 57)
        np.testing.assert_equal(f.starttime_second, 2)
        np.testing.assert_equal(f.patientcode.rstrip(), b'abcxyz99')
        np.testing.assert_equal(f.patientname.rstrip(), b'Hans Muller')
        np.testing.assert_equal(f.gender, b'Male')
        np.testing.assert_equal(f.birthdate, b'30 jun 1969')
        np.testing.assert_equal(f.patient_additional.rstrip(), b'patient')
        np.testing.assert_equal(f.admincode.rstrip(), b'Dr. X')
        np.testing.assert_equal(f.technician.rstrip(), b'Mr. Spotty')
        np.testing.assert_equal(f.recording_additional.rstrip(), b'unit test file')
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
        np.testing.assert_equal(f.getSignalFrequencies()[0], 200)

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