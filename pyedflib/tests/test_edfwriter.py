# -*- coding: utf-8 -*-
# Copyright (c) 2015 Holger Nahrstaedt
from __future__ import division, print_function, absolute_import

import os
import numpy as np
# from numpy.testing import (assert_raises, run_module_suite,
#                            assert_equal, assert_allclose, assert_almost_equal)
import unittest
import pyedflib
from datetime import datetime, date


class TestEdfWriter(unittest.TestCase):
    def setUp(self):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.bdfplus_data_file = os.path.join(data_dir, 'tmp_test_file_plus.bdf')
        self.edfplus_data_file = os.path.join(data_dir, 'tmp_test_file_plus.edf')
        self.bdf_data_file = os.path.join(data_dir, 'tmp_test_file.bdf')
        self.edf_data_file = os.path.join(data_dir, 'tmp_test_file.edf')

    def test_EdfWriter_BDFplus(self):
        channel_info1 = {'label': 'test_label', 'dimension': 'mV', 'sample_rate': 100,
                        'physical_max': 1.0, 'physical_min': -1.0,
                        'digital_max': 8388607, 'digital_min': -8388608,
                        'prefilter': 'pre1', 'transducer': 'trans1'}
        channel_info2 = {'label': 'test_label', 'dimension': 'mV', 'sample_rate': 100,
                             'physical_max': 1.0, 'physical_min': -1.0,
                            'digital_max': 8388607, 'digital_min': -8388608,
                            'prefilter': 'pre1', 'transducer': 'trans1'}
        f = pyedflib.EdfWriter(self.bdfplus_data_file, 2,
                               file_type=pyedflib.FILETYPE_BDFPLUS)
        f.setSignalHeader(0,channel_info1)
        f.setSignalHeader(1,channel_info2)
        f.setTechnician('tec1')
        f.setRecordingAdditional('recAdd1')
        f.setPatientName('pat1')
        f.setPatientCode('code1')
        f.setPatientAdditional('patAdd1')
        f.setAdmincode('admin1')
        f.setEquipment('eq1')
        f.setGender(1)
        f.setBirthdate(date(1951, 8, 2))
        f.setStartdatetime(datetime(2017, 1, 1, 1, 1, 1))
        f.setSamplefrequency(1,200)
        f.setPhysicalMaximum(1,2)
        f.setPhysicalMinimum(1,-2)
        f.setLabel(1,'test 2')
        f.setPhysicalDimension(1,'l2')
        f.setTransducer(1,'trans2')
        f.setPrefilter(1,'pre2')
        data1 = np.ones(100) * 0.1
        data2 = np.ones(200) * 0.2
        f.writePhysicalSamples(data1)
        f.writePhysicalSamples(data2)
        f.writePhysicalSamples(data1)
        f.writePhysicalSamples(data2)
        del f

        f = pyedflib.EdfReader(self.bdfplus_data_file)
        np.testing.assert_equal(f.getTechnician(), 'tec1')
        np.testing.assert_equal(f.getRecordingAdditional(), 'recAdd1')
        np.testing.assert_equal(f.getPatientName(), 'pat1')
        np.testing.assert_equal(f.getPatientCode(), 'code1')
        np.testing.assert_equal(f.getPatientAdditional(), 'patAdd1')
        np.testing.assert_equal(f.getAdmincode(), 'admin1')
        np.testing.assert_equal(f.getEquipment(), 'eq1')
        np.testing.assert_equal(f.getGender(), 'Male')
        np.testing.assert_equal(f.getBirthdate(), '02 aug 1951')
        np.testing.assert_equal(f.getStartdatetime(), datetime(2017, 1, 1, 1, 1, 1))

        np.testing.assert_equal(f.getLabel(0), 'test_label')
        np.testing.assert_equal(f.getPhysicalDimension(0), 'mV')
        np.testing.assert_equal(f.getPrefilter(0), 'pre1')
        np.testing.assert_equal(f.getTransducer(0), 'trans1')
        np.testing.assert_equal(f.getSampleFrequency(0), 100)

        np.testing.assert_equal(f.getLabel(1), 'test 2')
        np.testing.assert_equal(f.getPhysicalDimension(1), 'l2')
        np.testing.assert_equal(f.getPrefilter(1), 'pre2')
        np.testing.assert_equal(f.getTransducer(1), 'trans2')
        np.testing.assert_equal(f.getSampleFrequency(1), 200)
        np.testing.assert_equal(f.getPhysicalMaximum(1), 2)
        np.testing.assert_equal(f.getPhysicalMinimum(1), -2)
        del f

    def test_EdfWriter_BDFplus2(self):
        channel_info1 = {'label': 'test_label', 'dimension': 'mV', 'sample_rate': 100,
                        'physical_max': 1.0, 'physical_min': -1.0,
                        'digital_max': 8388607, 'digital_min': -8388608,
                        'prefilter': 'pre1', 'transducer': 'trans1'}
        channel_info2 = {'label': 'test_label', 'dimension': 'mV', 'sample_rate': 100,
                             'physical_max': 1.0, 'physical_min': -1.0,
                            'digital_max': 8388607, 'digital_min': -8388608,
                            'prefilter': 'pre1', 'transducer': 'trans1'}
        f = pyedflib.EdfWriter(self.bdfplus_data_file, 2,
                               file_type=pyedflib.FILETYPE_BDFPLUS)
        f.setSignalHeader(0,channel_info1)
        f.setSignalHeader(1,channel_info2)
        f.setTechnician('tec1')
        f.setRecordingAdditional('recAdd1')
        f.setPatientName('empty')

        f.setPatientCode('code1')
        f.setPatientAdditional('patAdd1')
        f.setAdmincode('admin1')
        f.setEquipment('eq1')
        f.setGender("Male")
        f.setBirthdate(date(1951, 8, 2))
        f.setStartdatetime(datetime(2017, 1, 1, 1, 1, 1))
        f.setSamplefrequency(1,100)
        f.setPhysicalMaximum(1,2)
        f.setPhysicalMinimum(1,-2)

        data1 = np.ones(100) * 0.1
        data2 = np.ones(100) * 0.2
        f.writePhysicalSamples(data1)
        f.writePhysicalSamples(data2)
        f.writePhysicalSamples(data2)
        f.writePhysicalSamples(data1)
        del f

        f = pyedflib.EdfReader(self.bdfplus_data_file)
        np.testing.assert_equal(f.getTechnician(), 'tec1')
        np.testing.assert_equal(f.getRecordingAdditional(), 'recAdd1')
        np.testing.assert_equal(f.getPatientName(), 'empty')
        np.testing.assert_equal(f.getPatientCode(), 'code1')
        np.testing.assert_equal(f.getPatientAdditional(), 'patAdd1')
        np.testing.assert_equal(f.getAdmincode(), 'admin1')
        np.testing.assert_equal(f.getEquipment(), 'eq1')
        np.testing.assert_equal(f.getGender(), 'Male')
        np.testing.assert_equal(f.getBirthdate(), '02 aug 1951')
        np.testing.assert_equal(f.getStartdatetime(), datetime(2017, 1, 1, 1, 1, 1))

        x01 = f.readSignal(0,000,100)
        x02 = f.readSignal(0,100,100)
        x11 = f.readSignal(1,000,100)
        x12 = f.readSignal(1,100,100)
        np.testing.assert_almost_equal(np.sum(np.abs(x01-data1)),0,decimal=4)
        np.testing.assert_almost_equal(np.sum(np.abs(x02-data2)),0,decimal=4)
        np.testing.assert_almost_equal(np.sum(np.abs(x11-data2)),0,decimal=4)
        np.testing.assert_almost_equal(np.sum(np.abs(x12-data1)),0,decimal=4)
        del f

    def test_EdfWriter_BDF(self):
        channel_info1 = {'label': 'test_label', 'dimension': 'mV', 'sample_rate': 100,
                        'physical_max': 1.0, 'physical_min': -1.0,
                        'digital_max': 8388607, 'digital_min': -8388608,
                        'prefilter': 'pre1', 'transducer': 'trans1'}
        channel_info2 = {'label': 'test_label', 'dimension': 'mV', 'sample_rate': 100,
                            'physical_max': 1.0, 'physical_min': -1.0,
                            'digital_max': 8388607, 'digital_min': -8388608,
                            'prefilter': 'pre1', 'transducer': 'trans1'}
        f = pyedflib.EdfWriter(self.bdf_data_file, 2,
                               file_type=pyedflib.FILETYPE_BDF)
        f.setSignalHeader(0,channel_info1)
        f.setSignalHeader(1,channel_info2)

        data = np.ones(100) * 0.1
        f.writePhysicalSamples(data)
        f.writePhysicalSamples(data)
        f.writePhysicalSamples(data)
        f.writePhysicalSamples(data)
        del f

        f = pyedflib.EdfReader(self.bdf_data_file)

        np.testing.assert_equal(f.getLabel(0), 'test_label')
        np.testing.assert_equal(f.getPhysicalDimension(0), 'mV')
        np.testing.assert_equal(f.getPrefilter(0), 'pre1')
        np.testing.assert_equal(f.getTransducer(0), 'trans1')
        np.testing.assert_equal(f.getSampleFrequency(0), 100)
        del f

    def test_EdfWriter_EDFplus(self):
        channel_info = {'label': 'test_label', 'dimension': 'mV', 'sample_rate': 100,
                        'physical_max': 1.0, 'physical_min': -1.0,
                        'digital_max': 32767, 'digital_min': -32768,
                        'prefilter': 'pre1', 'transducer': 'trans1'}
        f = pyedflib.EdfWriter(self.edfplus_data_file, 1,
                               file_type=pyedflib.FILETYPE_EDFPLUS)

        header = {'technician': 'tec1', 'recording_additional': 'recAdd1', 'patientname': 'pat1',
                  'patient_additional': 'patAdd1', 'patientcode': 'code1', 'equipment': 'eq1',
                  'admincode':'admin1','gender':1,'startdate':datetime(2017, 1, 1, 1, 1, 1),'birthdate':date(1951, 8, 2)}
        f.setHeader(header)
        f.setSignalHeader(0,channel_info)
        data = np.ones(100) * 0.1
        f.writePhysicalSamples(data)
        f.writePhysicalSamples(data)
        del f

        f = pyedflib.EdfReader(self.edfplus_data_file)
        np.testing.assert_equal(f.getTechnician(), 'tec1')
        np.testing.assert_equal(f.getRecordingAdditional(), 'recAdd1')
        np.testing.assert_equal(f.getPatientName(), 'pat1')
        np.testing.assert_equal(f.getPatientCode(), 'code1')
        np.testing.assert_equal(f.getEquipment(), 'eq1')
        np.testing.assert_equal(f.getPatientAdditional(), 'patAdd1')
        np.testing.assert_equal(f.getAdmincode(), 'admin1')
        np.testing.assert_equal(f.getGender(), 'Male')
        np.testing.assert_equal(f.getBirthdate(), '02 aug 1951')
        np.testing.assert_equal(f.getStartdatetime(), datetime(2017, 1, 1, 1, 1, 1))

        np.testing.assert_equal(f.getLabel(0), 'test_label')
        np.testing.assert_equal(f.getPhysicalDimension(0), 'mV')
        np.testing.assert_equal(f.getPrefilter(0), 'pre1')
        np.testing.assert_equal(f.getTransducer(0), 'trans1')
        np.testing.assert_equal(f.getSampleFrequency(0), 100)
        del f

    def test_EdfWriter_EDF(self):
        channel_info1 = {'label': 'test_label', 'dimension': 'mV', 'sample_rate': 100,
                        'physical_max': 1.0, 'physical_min': -1.0,
                        'digital_max': 32767, 'digital_min': -32768,
                        'prefilter': 'pre1', 'transducer': 'trans1'}
        channel_info2 = {'label': 'test_label', 'dimension': 'mV', 'sample_rate': 100,
                             'physical_max': 1.0, 'physical_min': -1.0,
                            'digital_max': 32767, 'digital_min': -32768,
                            'prefilter': 'pre1', 'transducer': 'trans1'}
        f = pyedflib.EdfWriter(self.edf_data_file, 2,
                               file_type=pyedflib.FILETYPE_EDF)
        f.setSignalHeader(0,channel_info1)
        f.setSignalHeader(1,channel_info2)
        data = np.ones(100) * 0.1
        f.writePhysicalSamples(data)
        f.writePhysicalSamples(data)
        del f

        f = pyedflib.EdfReader(self.edf_data_file)

        np.testing.assert_equal(f.getLabel(0), 'test_label')
        np.testing.assert_equal(f.getPhysicalDimension(0), 'mV')
        np.testing.assert_equal(f.getPrefilter(0), 'pre1')
        np.testing.assert_equal(f.getTransducer(0), 'trans1')
        np.testing.assert_equal(f.getSampleFrequency(0), 100)
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
        f = pyedflib.EdfWriter(self.bdfplus_data_file, 2,
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

        f = pyedflib.EdfReader(self.bdfplus_data_file)
        data1_read = f.readSignal(0)
        data2_read = f.readSignal(1)
        f._close
        np.testing.assert_equal(len(data1), len(data1_read))
        np.testing.assert_equal(len(data2), len(data2_read))
        np.testing.assert_almost_equal(data1, data1_read)
        np.testing.assert_almost_equal(data2, data2_read)

    def test_EdfWriter_EDF_contextmanager(self):
        channel_info1 = {'label': 'test_label', 'dimension': 'mV', 'sample_rate': 100,
                        'physical_max': 1.0, 'physical_min': -1.0,
                        'digital_max': 32767, 'digital_min': -32768,
                        'prefilter': 'pre1', 'transducer': 'trans1'}
        channel_info2 = {'label': 'test_label', 'dimension': 'mV', 'sample_rate': 100,
                             'physical_max': 1.0, 'physical_min': -1.0,
                            'digital_max': 32767, 'digital_min': -32768,
                            'prefilter': 'pre1', 'transducer': 'trans1'}
        with pyedflib.EdfWriter(self.edf_data_file, 2, file_type=pyedflib.FILETYPE_EDF) as f:
            f.setSignalHeader(0,channel_info1)
            f.setSignalHeader(1,channel_info2)
            data = np.ones(100) * 0.1
            f.writePhysicalSamples(data)
            f.writePhysicalSamples(data)

        with pyedflib.EdfReader(self.edf_data_file) as f:
            np.testing.assert_equal(f.getLabel(0), 'test_label')
            np.testing.assert_equal(f.getPhysicalDimension(0), 'mV')
            np.testing.assert_equal(f.getPrefilter(0), 'pre1')
            np.testing.assert_equal(f.getTransducer(0), 'trans1')
            np.testing.assert_equal(f.getSampleFrequency(0), 100)

    def test_SampleWritingContextManager(self):
        channel_info1 = {'label':'test_label1', 'dimension':'mV', 'sample_rate':100,
                         'physical_max':1.0,'physical_min':-1.0,
                         'digital_max':8388607,'digital_min':-8388608,
                         'prefilter':'pre1','transducer':'trans1'}
        channel_info2 = {'label':'test_label2', 'dimension':'mV', 'sample_rate':100,
                         'physical_max':1.0,'physical_min':-1.0,
                         'digital_max':8388607,'digital_min':-8388608,
                         'prefilter':'pre2','transducer':'trans2'}
        
        with pyedflib.EdfWriter(self.bdfplus_data_file, 2, file_type=pyedflib.FILETYPE_BDFPLUS) as f:       

            f.setSignalHeader(0,channel_info1)
            f.setSignalHeader(1,channel_info2)
            data1 = np.ones(500) * 0.1
            data2 = np.ones(500) * 0.2
            data_list = []
            data_list.append(data1)
            data_list.append(data2)
            f.writeSamples(data_list)

        with pyedflib.EdfReader(self.bdfplus_data_file) as f:
            data1_read = f.readSignal(0)
            data2_read = f.readSignal(1)

        with pyedflib.EdfReader(self.bdfplus_data_file) as f:
            data1_read = f.readSignal(0)
            data2_read = f.readSignal(1)

        np.testing.assert_equal(len(data1), len(data1_read))
        np.testing.assert_equal(len(data2), len(data2_read))
        np.testing.assert_almost_equal(data1, data1_read)
        np.testing.assert_almost_equal(data2, data2_read)

    def test_SampleWriting2(self):
        channel_info1 = {'label':'test_label1', 'dimension':'mV', 'sample_rate':100,
                         'physical_max':1.0,'physical_min':-1.0,
                         'digital_max':8388607,'digital_min':-8388608,
                         'prefilter':'pre1','transducer':'trans1'}
        channel_info2 = {'label':'test_label2', 'dimension':'mV', 'sample_rate':100,
                         'physical_max':1.0,'physical_min':-1.0,
                         'digital_max':8388607,'digital_min':-8388608,
                         'prefilter':'pre2','transducer':'trans2'}
        f = pyedflib.EdfWriter(self.bdfplus_data_file, 2,
                              file_type=pyedflib.FILETYPE_BDFPLUS)
        f.setSignalHeader(0,channel_info1)
        f.setSignalHeader(1,channel_info2)

        data1 = np.ones(500) * 0.1
        data2 = np.ones(500) * 0.2
        data_list = []
        data_list.append(data1)
        data_list.append(data2)
        f.writeSamples(data_list)
        del f

        f = pyedflib.EdfReader(self.bdfplus_data_file)
        data1_read = f.readSignal(0)
        data2_read = f.readSignal(1)
        del f
        np.testing.assert_equal(len(data1), len(data1_read))
        np.testing.assert_equal(len(data2), len(data2_read))
        np.testing.assert_almost_equal(data1, data1_read)
        np.testing.assert_almost_equal(data2, data2_read)
        
    def test_SampleWriting_digital(self):

        dmin, dmax = [0, 1024]
        pmin, pmax = [0, 1.0]
        channel_info1 = {'label':'test_label1', 'dimension':'mV', 'sample_rate':100,
                         'physical_max':pmax,'physical_min':pmin,
                         'digital_max':dmax,'digital_min':dmin,
                         'prefilter':'pre1','transducer':'trans1'}
        channel_info2 = {'label':'test_label2', 'dimension':'mV', 'sample_rate':100,
                         'physical_max':pmax,'physical_min':pmin,
                         'digital_max':dmax,'digital_min':dmin,
                         'prefilter':'pre2','transducer':'trans2'}


        f = pyedflib.EdfWriter(self.bdfplus_data_file, 2,
                              file_type=pyedflib.FILETYPE_EDFPLUS)
        f.setSignalHeader(0,channel_info1)
        f.setSignalHeader(1,channel_info2)

        data1 = np.arange(500, dtype=np.float)
        data2 = np.arange(500, dtype=np.float)
        data_list = []
        data_list.append(data1)
        data_list.append(data2)
        with  np.testing.assert_raises(TypeError):
            f.writeSamples(data_list, digital=True)
        f.close()
        del f    

        f = pyedflib.EdfWriter(self.bdfplus_data_file, 2,
                              file_type=pyedflib.FILETYPE_EDFPLUS)
        f.setSignalHeader(0,channel_info1)
        f.setSignalHeader(1,channel_info2)

        data1 = np.arange(500, dtype=np.int32)
        data2 = np.arange(500, dtype=np.int32)
        data_list = []
        data_list.append(data1)
        data_list.append(data2)
        f.writeSamples(data_list, digital=True)
        del f

        f = pyedflib.EdfReader(self.bdfplus_data_file)
        data1_read = (f.readSignal(0) - pmin)/((pmax-pmin)/(dmax-dmin)) # converting back to digital
        data2_read = (f.readSignal(1) - pmin)/((pmax-pmin)/(dmax-dmin)) # converting back to digital
        del f
    
        np.testing.assert_equal(len(data1), len(data1_read))
        np.testing.assert_equal(len(data2), len(data2_read))
        np.testing.assert_almost_equal(data1, data1_read)
        np.testing.assert_almost_equal(data2, data2_read)

    def test_TestRoundingEDF(self):
        channel_info1 = {'label':'test_label1', 'dimension':'mV', 'sample_rate':100,
                         'physical_max':1.0,'physical_min':-1.0,
                         'digital_max':32767,'digital_min':-32768,
                         'prefilter':'pre1','transducer':'trans1'}
        f = pyedflib.EdfWriter(self.edfplus_data_file, 1,
                              file_type=pyedflib.FILETYPE_EDFPLUS)
        f.setSignalHeader(0,channel_info1)

        time = np.linspace(0,5,500)
        data1 = np.sin(2*np.pi*1*time)
        data_list = []
        data_list.append(data1)
        f.writeSamples(data_list)
        del f

        f = pyedflib.EdfReader(self.edfplus_data_file)
        data1_read = f.readSignal(0)
        del f
        np.testing.assert_equal(len(data1), len(data1_read))
        np.testing.assert_almost_equal(data1, data1_read,decimal=4)

        f = pyedflib.EdfWriter(self.edfplus_data_file, 1,
                                   file_type=pyedflib.FILETYPE_EDFPLUS)
        f.setSignalHeader(0,channel_info1)

        data_list = []
        data_list.append(data1_read)
        f.writeSamples(data_list)
        del f

        f = pyedflib.EdfReader(self.edfplus_data_file)
        data2_read = f.readSignal(0)
        del f
        np.testing.assert_equal(len(data1), len(data2_read))
        np.testing.assert_almost_equal(data1, data2_read,decimal=4)
        np.testing.assert_almost_equal(data1_read, data2_read, decimal=4)

    def test_AnnotationWriting(self):
        channel_info = {'label': 'test_label', 'dimension': 'mV', 'sample_rate': 100,
                        'physical_max': 1.0, 'physical_min': -1.0,
                        'digital_max': 8388607, 'digital_min': -8388608,
                        'prefilter': 'pre1', 'transducer': 'trans1'}
        f = pyedflib.EdfWriter(self.bdf_data_file, 1,
                               file_type=pyedflib.FILETYPE_BDFPLUS)
        f.setSignalHeader(0,channel_info)
        data = np.ones(100) * 0.1
        f.writePhysicalSamples(data)
        f.writePhysicalSamples(data)
        f.writePhysicalSamples(data)
        f.writePhysicalSamples(data)
        f.writeAnnotation(1.23, 0.2, u"annotation1_ä")
        f.writeAnnotation(0.25, -1, u"annotation2_ü")
        f.writeAnnotation(1.25, 0, u"annotation3_ö")
        f.writeAnnotation(1.30, -1, u"annotation4_ß")
        del f
        f = pyedflib.EdfReader(self.bdf_data_file)
        ann_time, ann_duration, ann_text = f.readAnnotations()
        del f
        np.testing.assert_almost_equal(ann_time[0], 1.23)
        np.testing.assert_almost_equal(ann_duration[0], 0.2)
        np.testing.assert_equal(ann_text[0], "annotation1_..")
        np.testing.assert_almost_equal(ann_time[1], 0.25)
        np.testing.assert_almost_equal(ann_duration[1], -1)
        np.testing.assert_equal(ann_text[1], "annotation2_..")
        np.testing.assert_almost_equal(ann_time[2], 1.25)
        np.testing.assert_almost_equal(ann_duration[2], 0)
        np.testing.assert_equal(ann_text[2], "annotation3_..")
        np.testing.assert_almost_equal(ann_time[3], 1.30)
        np.testing.assert_almost_equal(ann_duration[3], -1)
        np.testing.assert_equal(ann_text[3], "annotation4_..")

    def test_AnnotationWritingUTF8(self):
        channel_info = {'label': 'test_label', 'dimension': 'mV', 'sample_rate': 100,
                        'physical_max': 1.0, 'physical_min': -1.0,
                        'digital_max': 8388607, 'digital_min': -8388608,
                        'prefilter': u'test', 'transducer': 'trans1'}
        f = pyedflib.EdfWriter(self.bdf_data_file, 1,
                               file_type=pyedflib.FILETYPE_BDFPLUS)
        f.setSignalHeader(0,channel_info)
        data = np.ones(100) * 0.1
        f.writePhysicalSamples(data)
        f.writePhysicalSamples(data)
        f.writePhysicalSamples(data)
        f.writePhysicalSamples(data)
        f.writeAnnotation(1.23, 0.2, u"Zähne")
        f.writeAnnotation(0.25, -1, u"Fuß")
        f.writeAnnotation(1.25, 0, u"abc")
        del f

        f = pyedflib.EdfReader(self.bdf_data_file)
        ann_time, ann_duration, ann_text = f.readAnnotations()
        del f
        np.testing.assert_almost_equal(ann_time[0], 1.23)
        np.testing.assert_almost_equal(ann_duration[0], 0.2)
        np.testing.assert_equal(ann_text[0], "Z..hne")
        np.testing.assert_almost_equal(ann_time[1], 0.25)
        np.testing.assert_almost_equal(ann_duration[1], -1)
        np.testing.assert_equal(ann_text[1], "Fu..")
        np.testing.assert_almost_equal(ann_time[2], 1.25)
        np.testing.assert_almost_equal(ann_duration[2], 0)
        np.testing.assert_equal(ann_text[2], "abc")

    def test_BytesChars(self):
        channel_info = {'label': b'test_label', 'dimension': b'mV', 'sample_rate': 100,
                        'physical_max': 1.0, 'physical_min': -1.0,
                        'digital_max': 8388607, 'digital_min': -8388608,
                        'prefilter': b'      ', 'transducer': b'trans1'}
        f = pyedflib.EdfWriter(self.bdf_data_file, 1,
                               file_type=pyedflib.FILETYPE_BDFPLUS)
        f.setSignalHeader(0,channel_info)
        data = np.ones(100) * 0.1
        f.writePhysicalSamples(data)
        f.writePhysicalSamples(data)
        f.writePhysicalSamples(data)
        f.writePhysicalSamples(data)
        f.writeAnnotation(1.23, 0.2, b'Zaehne')
        f.writeAnnotation(0.25, -1, b'Fuss')
        f.writeAnnotation(1.25, 0, b'abc')
        del f

        f = pyedflib.EdfReader(self.bdf_data_file)
        ann_time, ann_duration, ann_text = f.readAnnotations()
        del f
        np.testing.assert_almost_equal(ann_time[0], 1.23)
        np.testing.assert_almost_equal(ann_duration[0], 0.2)
        np.testing.assert_equal(ann_text[0], "Zaehne")
        np.testing.assert_almost_equal(ann_time[1], 0.25)
        np.testing.assert_almost_equal(ann_duration[1], -1)
        np.testing.assert_equal(ann_text[1], "Fuss")
        np.testing.assert_almost_equal(ann_time[2], 1.25)
        np.testing.assert_almost_equal(ann_duration[2], 0)
        np.testing.assert_equal(ann_text[2], "abc")


if __name__ == '__main__':
    # run_module_suite(argv=sys.argv)
    unittest.main()
