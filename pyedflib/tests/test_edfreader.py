# -*- coding: utf-8 -*-
# Copyright (c) 2015 Holger Nahrstaedt
from __future__ import division, print_function, absolute_import

import os
import numpy as np
from datetime import datetime
# from numpy.testing import (assert_raises, run_module_suite,
#                            assert_equal, assert_allclose, assert_almost_equal)
import unittest
import pyedflib


class TestEdfReader(unittest.TestCase):
    def setUp(self):
        #data_dir = os.path.join(os.getcwd(), 'data')
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.edf_data_file = os.path.join(data_dir, 'test_generator.edf')
        self.bdf_broken_file = os.path.join(data_dir, 'tmp_broken_file.bdf')
        self.edf_broken_file = os.path.join(data_dir, 'tmp_broken_file.edf')

    def test_EdfReader(self):
        try:
            f = pyedflib.EdfReader(self.edf_data_file)
        except IOError:
            print('cannot open', self.edf_data_file)
            return

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
        try:
            f = pyedflib.EdfReader(self.edf_data_file)
        except IOError:
            print('cannot open', self.edf_data_file)
            return
        datetimeSoll = datetime(2011,4,4,12,57,2)
        np.testing.assert_equal(f.getStartdatetime(),datetimeSoll)
        np.testing.assert_equal(f.getPatientCode(), 'abcxyz99')
        np.testing.assert_equal(f.getPatientName(), 'Hans Muller')
        np.testing.assert_equal(f.getGender(), 'Male')
        np.testing.assert_equal(f.getBirthdate(), '30 jun 1969')
        np.testing.assert_equal(f.getPatientAdditional(), 'patient')
        np.testing.assert_equal(f.getAdmincode(), 'Dr. X')
        np.testing.assert_equal(f.getTechnician(), 'Mr. Spotty')
        np.testing.assert_equal(f.getRecordingAdditional(), 'unit test file')
        np.testing.assert_equal(f.getFileDuration(), 600)
        fileHeader = f.getHeader()
        np.testing.assert_equal(fileHeader["patientname"], 'Hans Muller')
        f._close()
        del f

    def test_EdfReader_signalInfos(self):
        try:
            f = pyedflib.EdfReader(self.edf_data_file)
        except IOError:
            print('cannot open', self.edf_data_file)
            return
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

    def test_EdfReader_ReadAnnotations(self):
        try:
            f = pyedflib.EdfReader(self.edf_data_file, pyedflib.DO_NOT_READ_ANNOTATIONS)
        except IOError:
            print('cannot open', self.edf_data_file)
            return

        ann_index, ann_duration, ann_text = f.readAnnotations()
        np.testing.assert_equal(ann_index.size, 0)

        f._close()
        del f

        try:
            f = pyedflib.EdfReader(self.edf_data_file, pyedflib.READ_ALL_ANNOTATIONS)
        except IOError:
            print('cannot open', self.edf_data_file)
            return

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

    def test_EdfReader_Check_Size(self):
        channel_info = {'label': 'test_label', 'dimension': 'mV', 'sample_rate': 100,
                        'physical_max': 1.0, 'physical_min': -1.0,
                        'digital_max': 8388607, 'digital_min': -8388608,
                        'prefilter': 'pre1', 'transducer': 'trans1'}
        f = pyedflib.EdfWriter(self.bdf_broken_file, 1,file_type=pyedflib.FILETYPE_BDFPLUS)
        f.setSignalHeader(0,channel_info)
        f.setTechnician('tec1')
        data = np.ones(100) * 0.1
        f.writePhysicalSamples(data)
        f.writePhysicalSamples(data)
        f.close()
        del f

        f = pyedflib.EdfReader(self.bdf_broken_file, pyedflib.READ_ALL_ANNOTATIONS, pyedflib.DO_NOT_CHECK_FILE_SIZE)
        np.testing.assert_equal(f.getTechnician(), 'tec1')

        np.testing.assert_equal(f.getLabel(0), 'test_label')
        np.testing.assert_equal(f.getPhysicalDimension(0), 'mV')
        np.testing.assert_equal(f.getPrefilter(0), 'pre1')
        np.testing.assert_equal(f.getTransducer(0), 'trans1')
        np.testing.assert_equal(f.getSampleFrequency(0), 100)
        f._close()
        del f

if __name__ == '__main__':
    # run_module_suite(argv=sys.argv)
    unittest.main()
