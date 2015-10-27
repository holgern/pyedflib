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


class TestEdfWriter(unittest.TestCase):
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
        f.setSignalHeader(0,channel_info)
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

    def test_SampleWriting(self):
        channel_info1 = {'label':'test_label1', 'dimension':'mV', 'sample_rate':100,
                         'physical_max':1.0,'physical_min':-1.0,
                         'digital_max':8388607,'digital_min':-8388608,
                         'prefilter':'pre1','transducer':'trans1'}
        channel_info2 = {'label':'test_label2', 'dimension':'mV', 'sample_rate':100,
                         'physical_max':1.0,'physical_min':-1.0,
                         'digital_max':8388607,'digital_min':-8388608,
                         'prefilter':'pre2','transducer':'trans2'}
        f = pyedflib.EdfWriter(self.bdf_data_file, 2,
                              file_type=pyedflib.FILETYPE_BDFPLUS)
        f.setSignalHeader(0,channel_info1)
        f.setSignalHeader(1,channel_info2)

        data1 = np.ones(500) * 0.1
        data2 = np.ones(500) * 0.2
        data_list = []
        data_list.append(data1)
        data_list.append(data2)
        f.writeSamples(data_list)

        f.close()
        del f

        f = pyedflib.EdfReader(self.bdf_data_file)
        data1_read = f.readSignal(0)
        data2_read = f.readSignal(1)
        f._close()
        del f
        np.testing.assert_equal(len(data1), len(data1_read))
        np.testing.assert_equal(len(data2), len(data2_read))
        np.testing.assert_almost_equal(data1, data1_read)
        np.testing.assert_almost_equal(data2, data2_read)

    def test_AnnotationWriting(self):
        channel_info = {'label':'test_label', 'dimension':'mV', 'sample_rate':100,
                         'physical_max':1.0,'physical_min':-1.0,
                         'digital_max':8388607,'digital_min':-8388608,
                         'prefilter':'pre1','transducer':'trans1'}
        f = pyedflib.EdfWriter(self.bdf_data_file, 1,
                              file_type=pyedflib.FILETYPE_BDFPLUS)
        f.setSignalHeader(0,channel_info)
        data = np.ones(100) * 0.1
        f.writePhyisicalSamples(data)
        f.writePhyisicalSamples(data)
        f.writePhyisicalSamples(data)
        f.writePhyisicalSamples(data)
        f.writeAnnotation(1.23,0.2,"annotation1")
        f.writeAnnotation(0.25,-1,"annotation2")
        f.writeAnnotation(1.25,0,"annotation3")
        f.close()
        del f

        f = pyedflib.EdfReader(self.bdf_data_file)
        ann_time, ann_duration, ann_text = f.readAnnotations()
        f._close()
        del f
        np.testing.assert_almost_equal(ann_time[0], 1.23)
        np.testing.assert_almost_equal(ann_duration[0], 0.2)
        np.testing.assert_equal(ann_text[0], b"annotation1")
        np.testing.assert_almost_equal(ann_time[1], 0.25)
        np.testing.assert_almost_equal(ann_duration[1], -1)
        np.testing.assert_equal(ann_text[1], b"annotation2")
        np.testing.assert_almost_equal(ann_time[2], 1.25)
        np.testing.assert_almost_equal(ann_duration[2], 0)
        np.testing.assert_equal(ann_text[2], b"annotation3")


if __name__ == '__main__':
    # run_module_suite(argv=sys.argv)
    unittest.main()
