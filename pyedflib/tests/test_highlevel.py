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
        self.bdfplus_data_file = os.path.join(data_dir, 'tmp_test_file_plus.bdf')
        self.edfplus_data_file = os.path.join(data_dir, 'tmp_test_file_plus.edf')
        self.bdf_data_file = os.path.join(data_dir, 'tmp_test_file.bdf')
        self.edf_data_file = os.path.join(data_dir, 'tmp_test_file.edf')

    def test_read_write_edf(self):
        startdate = datetime.now()
        header = highlevel.make_header(technician='tech', recording_additional='radd',
                                                patientname='name', patient_additional='padd',
                                                patientcode='42', equipment='eeg', admincode='420',
                                                gender='male', startdate=startdate,birthdate='05.09.1980')
        annotations = [[50, -1, 'begin'],[150, -1, 'end']]
        header['annotations'] = annotations
        signal_headers = highlevel.make_signal_headers(['ch'+str(i) for i in range(5)])
        signals = np.random.rand(5, 256*300) #5 minutes of eeg
        
        success = highlevel.write_edf(self.edfplus_data_file,signals, signal_headers, header)
        self.assertTrue(os.path.isfile(self.edfplus_data_file))
        self.assertGreater(os.path.getsize(self.edfplus_data_file), 0)
        self.assertTrue(success)
        
    


if __name__ == '__main__':
    # run_module_suite(argv=sys.argv)
    unittest.main()
