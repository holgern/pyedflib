# -*- coding: utf-8 -*-
# Copyright (c) 2015 Holger Nahrstaedt
from __future__ import division, print_function, absolute_import

import os, sys
import numpy as np
# from numpy.testing import (assert_raises, run_module_suite,
#                            assert_equal, assert_allclose, assert_almost_equal)
import unittest
from pyedflib import highlevel
from datetime import datetime, date

class TestHighLevel(unittest.TestCase):
    
    def setUp(self):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.edfplus_data_file = os.path.join(data_dir, 'tmp_test_file_plus.edf')
        self.test_generator = os.path.join(data_dir, 'test_generator.edf')


    def test_dig2phys_calc(self):
        signals_phys, shead, _ = highlevel.read_edf(self.test_generator)
        signals_dig, _, _ = highlevel.read_edf(self.test_generator, digital=True)
                
        dmin, dmax = shead[0]['digital_min'],  shead[0]['digital_max']
        pmin, pmax = shead[0]['physical_min'],  shead[0]['physical_max']
        
        # convert to physical
        signal_phys2 = highlevel.dig2phys(signals_dig, dmin, dmax, pmin, pmax)        
        np.testing.assert_allclose(signals_phys, signal_phys2)
        
        # convert to digital
        signals_dig2 = highlevel.phys2dig(signals_phys, dmin, dmax, pmin, pmax)
        signals_dig2 = np.rint(signals_dig2)
        np.testing.assert_allclose(signals_dig, signals_dig2)

    def test_read_write_edf(self):
        startdate = datetime.now()
        t = startdate
        startdate = datetime(t.year,t.month,t.day,t.hour, t.minute,t.second)
        
        header = highlevel.make_header(technician='tech', recording_additional='radd',
                                                patientname='name', patient_additional='padd',
                                                patientcode='42', equipment='eeg', admincode='420',
                                                gender='Male', startdate=startdate,birthdate='05.09.1980')
        annotations = [[0.01, -1, 'begin'],[0.5, -1, 'middle'],[10, -1, 'end']]
        header['annotations'] = annotations
        signal_headers1 = highlevel.make_signal_headers(['ch'+str(i) for i in range(5)])
        signals = np.random.rand(5, 256*300)*200 #5 minutes of eeg
        
        success = highlevel.write_edf(self.edfplus_data_file, signals, signal_headers1, header)
        self.assertTrue(os.path.isfile(self.edfplus_data_file))
        self.assertGreater(os.path.getsize(self.edfplus_data_file), 0)
        self.assertTrue(success)
        
        signals2, signal_headers2, header2 = highlevel.read_edf(self.edfplus_data_file)

        self.assertEqual(len(signals2), 5)
        self.assertEqual(len(signals2), len(signal_headers2))
        for shead1, shead2 in zip(signal_headers1, signal_headers2):
            self.assertDictEqual(shead1, shead2)
            
        self.assertDictEqual(header, header2)
        np.testing.assert_allclose(signals, signals2, atol=0.01)
    
        signals = (signals*100).astype(np.int8)
        success = highlevel.write_edf(self.edfplus_data_file, signals,  signal_headers1, header, digital=True)
        self.assertTrue(os.path.isfile(self.edfplus_data_file))
        self.assertGreater(os.path.getsize(self.edfplus_data_file), 0)
        self.assertTrue(success)
        
        signals2, signal_headers2, header2 = highlevel.read_edf(self.edfplus_data_file, digital=True)

        self.assertEqual(len(signals2), 5)
        self.assertEqual(len(signals2), len(signal_headers2))
        for shead1, shead2 in zip(signal_headers1, signal_headers2):
            self.assertDictEqual(shead1, shead2)
            
        self.assertDictEqual(header, header2)
        np.testing.assert_array_equal(signals, signals2)
        
        
    def test_read_write_with_annotations(self):
        signals, signal_headers, header = highlevel.read_edf(self.test_generator)
        expected = [[0.0, -1, 'Recording starts'], [600.0, -1, 'Recording ends']]
        self.assertEqual(header['annotations'], expected)
        
        highlevel.write_edf(self.edfplus_data_file, signals, signal_headers, header)
        signals2, signal_header2s, header2 = highlevel.read_edf(self.edfplus_data_file)
        self.assertEqual(header['annotations'], header2['annotations'])

        
    def test_quick_write(self):
        signals = np.random.randint(-2048, 2048, [3, 256*60])
        highlevel.write_edf_quick(self.edfplus_data_file, signals.astype(np.int32), sfreq=256, digital=True)
        signals2, _, _ = highlevel.read_edf(self.edfplus_data_file, digital=True)
        np.testing.assert_allclose(signals, signals2)
        signals = np.random.rand(3, 256*60)
        highlevel.write_edf_quick(self.edfplus_data_file, signals, sfreq=256)
        signals2, _, _ = highlevel.read_edf(self.edfplus_data_file)
        np.testing.assert_allclose(signals, signals2, atol=0.00002)
        
    def test_read_write_diff_sfreq(self):
        
        signals = []
        sfreqs = [1, 64, 128, 200]
        sheaders = []
        for sfreq in sfreqs:
            signals.append(np.random.randint(-2048, 2048, sfreq*60))
            shead = highlevel.make_signal_header('ch{}'.format(sfreq), sample_rate=sfreq)
            sheaders.append(shead)
        highlevel.write_edf(self.edfplus_data_file, signals, sheaders, digital=True)
        signals2, sheaders2, _ = highlevel.read_edf(self.edfplus_data_file, digital=True)
        for s1, s2 in zip(signals, signals2):
            np.testing.assert_allclose(s1, s2)
        
            
if __name__ == '__main__':
    # run_module_suite(argv=sys.argv)
    unittest.main()
