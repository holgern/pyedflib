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


class PLATest(unittest.TestCase):
    def setUp(self):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.edf_data_file = os.path.join(data_dir, 'test_generator.edf')

    def test_EdfReader(self):
        f = pyedflib.edfreader.EdfReader(self.edf_data_file)
        ann_index, ann_duration, ann_text = f.readAnnotations()
        self.assertAlmostEqual(ann_index[0], 0)
        self.assertAlmostEqual(ann_index[1], 600)

        self.assertEqual(f.signals_in_file, 11)
        self.assertEqual(f.datarecords_in_file, 600)

        for i in np.arange(11):
            self.assertAlmostEqual(f.getSignalFrequencies()[i], 200)
            self.assertEqual(f.getNSamples()[i], 120000)

        f._close()
        del f

    def test_EdfReader_headerInfos(self):
        f = pyedflib.edfreader.EdfReader(self.edf_data_file)
        self.assertEqual(f.startdate_day, 4)
        self.assertEqual(f.startdate_month, 4)
        self.assertEqual(f.startdate_year, 2011)
        self.assertEqual(f.starttime_hour, 12)
        self.assertEqual(f.starttime_minute, 57)
        self.assertEqual(f.starttime_second, 2)
        self.assertEqual(f.patientcode.rstrip(), b'abcxyz99')
        self.assertEqual(f.patientname.rstrip(), b'Hans Muller')
        self.assertEqual(f.gender, b'Male')
        self.assertEqual(f.birthdate, b'30 jun 1969')
        self.assertEqual(f.patient_additional.rstrip(), b'patient')
        self.assertEqual(f.admincode.rstrip(), b'Dr. X')
        self.assertEqual(f.technician.rstrip(), b'Mr. Spotty')
        self.assertEqual(f.recording_additional.rstrip(), b'unit test file')
        f._close()
        del f

    def test_EdfReader_signalInfos(self):
        f = pyedflib.edfreader.EdfReader(self.edf_data_file)
        self.assertEqual(f.getSignalLabels()[0], b'squarewave')
        self.assertEqual(f.getLabel(0), b'squarewave')
        self.assertEqual(f.getPhysicalDimension(0), b'uV')
        self.assertEqual(f.getPrefilter(0), b'pre1')
        self.assertEqual(f.getTransducer(0), b'trans1')
        self.assertEqual(f.getSampleFrequency(0), 200)
        self.assertEqual(f.getSignalFrequencies()[0], 200)

        self.assertEqual(f.getSignalLabels()[1], b'ramp')
        self.assertEqual(f.getSignalLabels()[2], b'pulse')
        self.assertEqual(f.getSignalLabels()[3], b'noise')
        self.assertEqual(f.getSignalLabels()[4], b'sine 1 Hz')
        self.assertEqual(f.getSignalLabels()[5], b'sine 8 Hz')
        f._close()
        del f

if __name__ == '__main__':
    # run_module_suite(argv=sys.argv)
    unittest.main()