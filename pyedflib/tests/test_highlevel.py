# -*- coding: utf-8 -*-
# Copyright (c) 2015 Holger Nahrstaedt
from __future__ import division, print_function, absolute_import

import os
import numpy as np
# from numpy.testing import (assert_raises, run_module_suite,
#                            assert_equal, assert_allclose, assert_almost_equal)
import unittest
from pyedflib import highlevel
from datetime import datetime, date


class TestEdfWriter(unittest.TestCase):
    def setUp(self):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.edfplus_data_file = os.path.join(data_dir, 'tmp_test_file_plus.edf')

    def test_read_write_edf(self):
        startdate = datetime.now()
        header = highlevel.make_header(technician='tech', recording_additional='radd',
                                                patientname='name', patient_additional='padd',
                                                patientcode='42', equipment='eeg', admincode='420',
                                                gender='Male', startdate=startdate,birthdate='05.09.1980')
        annotations = [[50, -1, 'begin'],[150, -1, 'end']]
        header['annotations'] = annotations
        signal_headers1 = highlevel.make_signal_headers(['ch'+str(i) for i in range(5)])
        signals = np.random.rand(5, 256*300)*200 #5 minutes of eeg
        
        success = highlevel.write_edf(self.edfplus_data_file, signals, signal_headers1, header)
        self.assertTrue(os.path.isfile(self.edfplus_data_file))
        self.assertGreater(os.path.getsize(self.edfplus_data_file), 0)
        self.assertTrue(success)
        
        signals2, signal_headers2, header2 = highlevel.read_edf(self.edfplus_data_file)
        t = header['startdate']
        header['startdate'] = datetime(t.year,t.month,t.day,t.hour, t.minute,t.second)
        
        self.assertEqual(len(signals2), 5)
        self.assertEqual(len(signals2), len(signal_headers2))
        for shead1, shead2 in zip(signal_headers1, signal_headers2):
            self.assertDictEqual(shead1, shead2)
        self.assertEqual(header, header2)
        np.testing.assert_allclose(signals, signals2, atol=0.01)
    
        signals = (signals*100).astype(np.int8)
        success = highlevel.write_edf(self.edfplus_data_file, signals,  signal_headers1, header, digital=True)
        self.assertTrue(os.path.isfile(self.edfplus_data_file))
        self.assertGreater(os.path.getsize(self.edfplus_data_file), 0)
        self.assertTrue(success)
        
        signals2, signal_headers2, header2 = highlevel.read_edf(self.edfplus_data_file, digital=True)
        t = header['startdate']
        header['startdate'] = datetime(t.year, t.month, t.day, t.hour, t.minute, t.second)
        
        self.assertEqual(len(signals2), 5)
        self.assertEqual(len(signals2), len(signal_headers2))
        for shead1, shead2 in zip(signal_headers1, signal_headers2):
            self.assertDictEqual(shead1, shead2)
        self.assertEqual(header, header2)
        np.testing.assert_array_equal(signals, signals2)

if __name__ == '__main__':
    # run_module_suite(argv=sys.argv)
    unittest.main()
