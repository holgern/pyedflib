# -*- coding: utf-8 -*-
# Copyright (c) 2019 - 2021 Simon Kern
# Copyright (c) 2015 Holger Nahrstaedt

import os
import numpy as np
# from numpy.testing import (assert_raises, run_module_suite,
#                            assert_equal, assert_allclose, assert_almost_equal)
import unittest
import pyedflib
from pyedflib.edfwriter import EdfWriter, ChannelDoesNotExist, WrongInputSize
from pyedflib.edfreader import EdfReader
from datetime import datetime, date


class TestEdfWriter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        cls.bdfplus_data_file = os.path.join(data_dir, 'tmp_test_file_plus.bdf')
        cls.edfplus_data_file = os.path.join(data_dir, 'tmp_test_file_plus.edf')
        cls.bdf_data_file = os.path.join(data_dir, 'tmp_test_file.bdf')
        cls.edf_data_file = os.path.join(data_dir, 'tmp_test_file.edf')
        cls.data_dir = data_dir

    @classmethod
    def tearDownClass(cls):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        tmpfiles = [f for f in os.listdir(data_dir) if f.startswith('tmp')]
        for file in tmpfiles:
            try:
                os.remove(os.path.join(data_dir, file))
            except Exception as e:
                print(e)
        
    def test_exceptions_raised(self):
        
        n_channels = 5
        f = pyedflib.EdfWriter(self.edfplus_data_file, n_channels,
                              file_type=pyedflib.FILETYPE_EDFPLUS)
        
        
        functions_ch = [f.setSamplefrequency,
                        f.setSignalHeader,
                        f.setPhysicalMaximum,
                        f.setPhysicalMinimum,
                        f.setDigitalMaximum,
                        f.setDigitalMinimum,
                        f.setLabel,
                        f.setTransducer,
                        f.setPrefilter]
        for func in functions_ch:
            with self.assertRaises(ChannelDoesNotExist):
                func(-1, None)
            with self.assertRaises(ChannelDoesNotExist):
                f.setSignalHeader(n_channels+1, None)    
                
    def test_write_functions(self):
        channel_info1 = {'label': 'label1', 'dimension': 'mV', 'sample_frequency': 100,
                        'physical_max': 32767, 'physical_min': -32768,
                        'digital_max': 32767, 'digital_min': -32768,
                        'prefilter': 'pre1', 'transducer': 'trans1'}
        channel_info2 = {'label': 'label2', 'dimension': 'mV', 'sample_frequency': 100,
                              'physical_max': 32767, 'physical_min': -32768,
                            'digital_max': 32767, 'digital_min': -32768,
                            'prefilter': 'pre1', 'transducer': 'trans1'}

        # I'm not raising the errors, but rather go through all tests and
        # raise the error at the end if there was any.
        # this makes it easier to find patterns of which functions fail generally
        error = False

        print() # empty line for readability

        # just looping through all write methods and see if they work
        for file_type in [0, 1, 2, 3]:
            filename = os.path.join(self.data_dir, 'tmp_write_{}.edf'.format(file_type))

            with pyedflib.EdfWriter(filename, 2,
                                file_type=file_type) as f:
                f.setSignalHeader(0, channel_info1)
                f.setSignalHeader(1, channel_info2)
                data = np.random.randint(-32768, 32767, 100)


                for i in range(2):
                    res = f.writePhysicalSamples(data.astype(float))
                    if res<0:
                        print(res, 'Error for filetype {} on writePhysicalSamples signal {}'.format(file_type, i))
                        error = True
                for i in range(2):
                    res = f.writeDigitalSamples(data.astype(np.int32))
                    if res<0:
                        print(res, 'Error for filetype {} on writeDigitalSamples signal {}'.format(file_type, i))
                        error = True

                res = f.blockWritePhysicalSamples(np.hstack([data.astype(float)]*2))
                if res<0:
                    print(res, 'Error for filetype {} on blockWritePhysicalSamples signal {}'.format(file_type, i))
                    error = True

                res = f.blockWriteDigitalSamples(np.hstack([data.astype(np.int32)]*2))
                if res<0:
                    print(res, 'Error for filetype {} on blockWriteDigitalSamples signal {}'.format(file_type, i))
                    error = True

            with pyedflib.EdfReader(filename) as f:
                data1 = f.readSignal(0)
                data2 = f.readSignal(1)
                try:
                    np.testing.assert_array_almost_equal(data1, data2)
                    self.assertEqual(data1.sum(), data.sum()*4, 'data written is not equal to data read')
                    self.assertEqual(len(data1), 400, 'didnt write 400 samples')
                except Exception as e:
                    print(e)
                    error=True

        if error:
            raise IOError('Writetests not successfully, see log for details')


    def test_subsecond_starttime(self):

        f = pyedflib.EdfWriter(self.edfplus_data_file, 1,
                                file_type=pyedflib.FILETYPE_EDFPLUS)

        channel_info = {'label': 'test_label', 'dimension': 'mV', 'sample_frequency': 100,
                        'physical_max': 1.0, 'physical_min': -1.0,
                        'digital_max': 32767, 'digital_min': -32768,
                        'prefilter': 'pre1', 'transducer': 'trans1'}
        startdate = datetime(2017, 1, 2, 13, 14, 15, 250)
        header = {'technician': 'tec1', 'recording_additional': 'recAdd1', 'patientname': 'pat1',
                  'patient_additional': 'patAdd1', 'patientcode': 'code1', 'equipment': 'eq1',
                  'admincode':'admin1','gender':1,'startdate':startdate,'birthdate':date(1951, 8, 2)}
        f.setHeader(header)
        f.setStartdatetime(startdate)
        f.setSignalHeader(0, channel_info)
        data = np.ones(100) * 0.1
        assert f.writePhysicalSamples(data)==0, 'error while writing physical sample'
        assert f.writePhysicalSamples(data)==0, 'error while writing physical sample'
        del f

        f = pyedflib.EdfReader(self.edfplus_data_file)
        startdate2 = f.getStartdatetime()
        assert startdate2==startdate, 'write {} != read {}'.format(startdate, startdate2)
        del f


    def test_subsecond_annotation(self):
        channel_info = {'label': 'test_label', 'dimension': 'mV', 'sample_frequency': 100,
                        'physical_max': 1.0, 'physical_min': -1.0,
                        'digital_max': 8388607, 'digital_min': -8388608,
                        'prefilter': 'pre1', 'transducer': 'trans1'}
        f = pyedflib.EdfWriter(self.bdfplus_data_file, 1,
                               file_type=pyedflib.FILETYPE_BDFPLUS)
        f.setSignalHeader(0,channel_info)
        data = np.ones(100) * 0.1
        f.writePhysicalSamples(data)
        f.writePhysicalSamples(data)
        f.writePhysicalSamples(data)
        f.writePhysicalSamples(data)
        f.writeAnnotation(1.23456, 0.2222, u"annotation1_ä")
        f.writeAnnotation(0.2567, -1, u"annotation2_ü")
        f.writeAnnotation(1.2567, 0, u"annotation3_ö")
        f.writeAnnotation(1.3067, -1, u"annotation4_ß")
        del f

        f = pyedflib.EdfReader(self.bdfplus_data_file)
        self.assertEqual(f.filetype, pyedflib.FILETYPE_BDFPLUS)

        ann_time, ann_duration, ann_text = f.readAnnotations()
        del f
        np.testing.assert_almost_equal(ann_time[0], 1.2345, decimal=4)
        np.testing.assert_almost_equal(ann_duration[0], 0.2222, decimal=4)
        np.testing.assert_equal(ann_text[0][0:12], "annotation1_")
        np.testing.assert_almost_equal(ann_time[1], 0.2567, decimal=4)
        np.testing.assert_almost_equal(ann_duration[1], -1)
        np.testing.assert_equal(ann_text[1][0:12], "annotation2_")
        np.testing.assert_almost_equal(ann_time[2], 1.2567, decimal=4)
        np.testing.assert_almost_equal(ann_duration[2], 0)
        np.testing.assert_equal(ann_text[2][0:12], "annotation3_")
        np.testing.assert_almost_equal(ann_time[3], 1.3067, decimal=4)
        np.testing.assert_almost_equal(ann_duration[3], -1)
        np.testing.assert_equal(ann_text[3][0:12], "annotation4_")

    def test_EdfWriter_BDFplus(self):
        channel_info1 = {'label': 'test_label', 'dimension': 'mV', 'sample_frequency': 100,
                        'physical_max': 1.0, 'physical_min': -1.0,
                        'digital_max': 8388607, 'digital_min': -8388608,
                        'prefilter': 'pre1', 'transducer': 'trans1'}
        channel_info2 = {'label': 'test_label', 'dimension': 'mV', 'sample_frequency': 100,
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
        # f.setBirthdate('2.8.1951')
        startdate = datetime(2017, 1, 1, 1, 1, 1)
        f.setStartdatetime(startdate)
        f.setStartdatetime(startdate.strftime("%d %b %Y %H:%M:%S"))
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
        self.assertEqual(f.filetype, pyedflib.FILETYPE_BDFPLUS)
        del f

    def test_EdfWriter_BDFplus2(self):
        channel_info1 = {'label': 'test_label', 'dimension': 'mV', 'sample_frequency': 100,
                        'physical_max': 1.0, 'physical_min': -1.0,
                        'digital_max': 8388607, 'digital_min': -8388608,
                        'prefilter': 'pre1', 'transducer': 'trans1'}
        channel_info2 = {'label': 'test_label', 'dimension': 'mV', 'sample_frequency': 100,
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
        self.assertEqual(f.filetype, pyedflib.FILETYPE_BDFPLUS)

        del f

    def test_EdfWriter_BDF(self):
        channel_info1 = {'label': 'test_label', 'dimension': 'mV', 'sample_frequency': 100,
                        'physical_max': 1.0, 'physical_min': -1.0,
                        'digital_max': 8388607, 'digital_min': -8388608,
                        'prefilter': 'pre1', 'transducer': 'trans1'}
        channel_info2 = {'label': 'test_label', 'dimension': 'mV', 'sample_frequency': 100,
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
        self.assertEqual(f.filetype, pyedflib.FILETYPE_BDF)
        del f

    def test_EdfWriter_EDFplus(self):
        channel_info = {'label': 'test_label', 'dimension': 'mV', 'sample_frequency': 100,
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
        assert f.writePhysicalSamples(data)==0, 'error while writing physical sample'
        assert f.writePhysicalSamples(data)==0, 'error while writing physical sample'
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
        self.assertEqual(f.filetype, pyedflib.FILETYPE_EDFPLUS)
        del f


    def test_EdfWriter_EDF(self):
        channel_info1 = {'label': 'test_label', 'dimension': 'mV', 'sample_frequency': 100,
                        'physical_max': 1.0, 'physical_min': -1.0,
                        'digital_max': 32767, 'digital_min': -32768,
                        'prefilter': 'pre1', 'transducer': 'trans1'}
        channel_info2 = {'label': 'test_label', 'dimension': 'mV', 'sample_frequency': 100,
                              'physical_max': 1.0, 'physical_min': -1.0,
                            'digital_max': 32767, 'digital_min': -32768,
                            'prefilter': 'pre1', 'transducer': 'trans1'}
        f = pyedflib.EdfWriter(self.edf_data_file, 2,
                                file_type=pyedflib.FILETYPE_EDF)
        f.setSignalHeader(0,channel_info1)
        f.setSignalHeader(1,channel_info2)
        data = np.ones(100) * 0.1
        assert f.writePhysicalSamples(data)==0, 'error while writing physical sample'
        assert f.writePhysicalSamples(data)==0, 'error while writing physical sample'
        del f

        f = pyedflib.EdfReader(self.edf_data_file)


        np.testing.assert_equal(f.getLabel(0), 'test_label')
        np.testing.assert_equal(f.getPhysicalDimension(0), 'mV')
        np.testing.assert_equal(f.getPrefilter(0), 'pre1')
        np.testing.assert_equal(f.getTransducer(0), 'trans1')
        np.testing.assert_equal(f.getSampleFrequency(0), 100)
        self.assertEqual(f.filetype, pyedflib.FILETYPE_EDF)

        del f


    def test_SampleWriting(self):
        channel_info1 = {'label':'test_label1', 'dimension':'mV', 'sample_frequency':100,
                          'physical_max':1.0,'physical_min':-1.0,
                          'digital_max':8388607,'digital_min':-8388608,
                          'prefilter':'pre1','transducer':'trans1'}
        channel_info2 = {'label':'test_label2', 'dimension':'mV', 'sample_frequency':100,
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
        self.assertEqual(f.filetype, pyedflib.FILETYPE_BDFPLUS)

    def test_EdfWriter_EDF_contextmanager(self):
        channel_info1 = {'label': 'test_label', 'dimension': 'mV', 'sample_frequency': 100,
                        'physical_max': 1.0, 'physical_min': -1.0,
                        'digital_max': 32767, 'digital_min': -32768,
                        'prefilter': 'pre1', 'transducer': 'trans1'}
        channel_info2 = {'label': 'test_label', 'dimension': 'mV', 'sample_frequency': 100,
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
            self.assertEqual(f.filetype, pyedflib.FILETYPE_EDF)

    def test_SampleWritingContextManager(self):
        channel_info1 = {'label':'test_label1', 'dimension':'mV', 'sample_frequency':100,
                          'physical_max':1.0,'physical_min':-1.0,
                          'digital_max':8388607,'digital_min':-8388608,
                          'prefilter':'pre1','transducer':'trans1'}
        channel_info2 = {'label':'test_label2', 'dimension':'mV', 'sample_frequency':100,
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
            self.assertEqual(f.filetype, pyedflib.FILETYPE_BDFPLUS)


        np.testing.assert_equal(len(data1), len(data1_read))
        np.testing.assert_equal(len(data2), len(data2_read))
        np.testing.assert_almost_equal(data1, data1_read)
        np.testing.assert_almost_equal(data2, data2_read)

    def test_SampleWriting2(self):
        channel_info1 = {'label':'test_label1', 'dimension':'mV', 'sample_frequency':100,
                          'physical_max':1.0,'physical_min':-1.0,
                          'digital_max':8388607,'digital_min':-8388608,
                          'prefilter':'pre1','transducer':'trans1'}
        channel_info2 = {'label':'test_label2', 'dimension':'mV', 'sample_frequency':100,
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
        self.assertEqual(f.filetype, pyedflib.FILETYPE_BDFPLUS)

        del f
        np.testing.assert_equal(len(data1), len(data1_read))
        np.testing.assert_equal(len(data2), len(data2_read))
        np.testing.assert_almost_equal(data1, data1_read)
        np.testing.assert_almost_equal(data2, data2_read)

    def test_SampleWriting_digital(self):

        dmin, dmax = [0, 1024]
        pmin, pmax = [0, 1.0]
        channel_info1 = {'label':'test_label1', 'dimension':'mV', 'sample_frequency':100,
                          'physical_max':pmax,'physical_min':pmin,
                          'digital_max':dmax,'digital_min':dmin,
                          'prefilter':'pre1','transducer':'trans1'}
        channel_info2 = {'label':'test_label2', 'dimension':'mV', 'sample_frequency':100,
                          'physical_max':pmax,'physical_min':pmin,
                          'digital_max':dmax,'digital_min':dmin,
                          'prefilter':'pre2','transducer':'trans2'}


        f = pyedflib.EdfWriter(self.edfplus_data_file, 2,
                              file_type=pyedflib.FILETYPE_EDFPLUS)
        f.setSignalHeader(0,channel_info1)
        f.setSignalHeader(1,channel_info2)

        data1 = np.arange(500, dtype=float)
        data2 = np.arange(500, dtype=float)
        data_list = []
        data_list.append(data1)
        data_list.append(data2)
        with  np.testing.assert_raises(TypeError):
            f.writeSamples(data_list, digital=True)
        f.close()
        del f

        f = pyedflib.EdfWriter(self.edfplus_data_file, 2,
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

        f = pyedflib.EdfReader(self.edfplus_data_file)
        data1_read = (f.readSignal(0) - pmin)/((pmax-pmin)/(dmax-dmin)) # converting back to digital
        data2_read = (f.readSignal(1) - pmin)/((pmax-pmin)/(dmax-dmin)) # converting back to digital
        self.assertEqual(f.filetype, pyedflib.FILETYPE_EDFPLUS)
        del f

        np.testing.assert_equal(len(data1), len(data1_read))
        np.testing.assert_equal(len(data2), len(data2_read))
        np.testing.assert_almost_equal(data1, data1_read)
        np.testing.assert_almost_equal(data2, data2_read)

    def test_TestRoundingEDF(self):
        channel_info1 = {'label':'test_label1', 'dimension':'mV', 'sample_frequency':100,
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
        self.assertEqual(f.filetype, pyedflib.FILETYPE_EDFPLUS)

        del f
        np.testing.assert_equal(len(data1), len(data2_read))
        np.testing.assert_almost_equal(data1, data2_read,decimal=4)
        np.testing.assert_almost_equal(data1_read, data2_read, decimal=4)

    def test_AnnotationWriting(self):
        channel_info = {'label': 'test_label', 'dimension': 'mV', 'sample_frequency': 100,
                        'physical_max': 1.0, 'physical_min': -1.0,
                        'digital_max': 8388607, 'digital_min': -8388608,
                        'prefilter': 'pre1', 'transducer': 'trans1'}
        f = pyedflib.EdfWriter(self.bdfplus_data_file, 1,
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
        f = pyedflib.EdfReader(self.bdfplus_data_file)
        self.assertEqual(f.filetype, pyedflib.FILETYPE_BDFPLUS)

        ann_time, ann_duration, ann_text = f.readAnnotations()
        del f
        np.testing.assert_almost_equal(ann_time[0], 1.23)
        np.testing.assert_almost_equal(ann_duration[0], 0.2)
        np.testing.assert_equal(ann_text[0][0:12], "annotation1_")
        np.testing.assert_almost_equal(ann_time[1], 0.25)
        np.testing.assert_almost_equal(ann_duration[1], -1)
        np.testing.assert_equal(ann_text[1][0:12], "annotation2_")
        np.testing.assert_almost_equal(ann_time[2], 1.25)
        np.testing.assert_almost_equal(ann_duration[2], 0)
        np.testing.assert_equal(ann_text[2][0:12], "annotation3_")
        np.testing.assert_almost_equal(ann_time[3], 1.30)
        np.testing.assert_almost_equal(ann_duration[3], -1)
        np.testing.assert_equal(ann_text[3][0:12], "annotation4_")

    def test_AnnotationWritingUTF8(self):
        channel_info = {'label': 'test_label', 'dimension': 'mV', 'sample_frequency': 100,
                        'physical_max': 1.0, 'physical_min': -1.0,
                        'digital_max': 8388607, 'digital_min': -8388608,
                        'prefilter': u'test', 'transducer': 'trans1'}
        f = pyedflib.EdfWriter(self.bdfplus_data_file, 1,
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

        f = pyedflib.EdfReader(self.bdfplus_data_file)
        self.assertEqual(f.filetype, pyedflib.FILETYPE_BDFPLUS)

        ann_time, ann_duration, ann_text = f.readAnnotations()
        del f
        np.testing.assert_almost_equal(ann_time[0], 1.23)
        np.testing.assert_almost_equal(ann_duration[0], 0.2)
        np.testing.assert_equal(ann_text[0][:1], "Z")
        np.testing.assert_equal(ann_text[0][3:], "hne")
        np.testing.assert_almost_equal(ann_time[1], 0.25)
        np.testing.assert_almost_equal(ann_duration[1], -1)
        np.testing.assert_equal(ann_text[1][:2], "Fu")
        np.testing.assert_almost_equal(ann_time[2], 1.25)
        np.testing.assert_almost_equal(ann_duration[2], 0)
        np.testing.assert_equal(ann_text[2], "abc")

    def test_BytesChars(self):
        channel_info = {'label': b'test_label', 'dimension': b'mV', 'sample_frequency': 100,
                        'physical_max': 1.0, 'physical_min': -1.0,
                        'digital_max': 8388607, 'digital_min': -8388608,
                        'prefilter': b'      ', 'transducer': b'trans1'}
        f = pyedflib.EdfWriter(self.bdfplus_data_file, 1,
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


        f = pyedflib.EdfReader(self.bdfplus_data_file)
        self.assertEqual(f.filetype, pyedflib.FILETYPE_BDFPLUS)

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

    def test_physical_range_inequality(self):
        # Prepare data
        channel_data1 = np.sin(np.arange(1,1001))

        channel_info1 = {'label': 'test_label_sin', 'dimension': 'mV', 'sample_frequency': 100,
                        'physical_max': 1, 'physical_min': -1,
                        'digital_max': 8388607, 'digital_min': -8388608,
                        'prefilter': 'pre1', 'transducer': 'trans1'}

        channel_data2 = np.zeros((1000,))
        channel_info2 = {'label': 'test_label_zero', 'dimension': 'mV', 'sample_frequency': 100,
                            'physical_max': max(channel_data2), 'physical_min': min(channel_data2),
                            'digital_max': 8388607, 'digital_min': -8388608,
                            'prefilter': 'pre1', 'transducer': 'trans1'}
        f = pyedflib.EdfWriter(self.edf_data_file, 2,
                                file_type=pyedflib.FILETYPE_BDF)
        f.setSignalHeader(0,channel_info1)
        f.setSignalHeader(1,channel_info2)

        # Test that assertion fails
        self.assertRaises(AssertionError, f.writeSamples, [channel_data1, channel_data2])


    def test_gender_setting_correctly(self):
        channel_info1 = {'label': 'test_label1', 'dimension': 'mV', 'sample_frequency': 100,
                         'physical_max': 3.0, 'physical_min': -3.0,
                         'digital_max': 32767, 'digital_min': -32768,
                         'prefilter': 'pre1', 'transducer': 'trans1'}
        channel_info2 = {'label': 'test_label2', 'dimension': 'mV', 'sample_frequency': 100,
                         'physical_max': 3.0, 'physical_min': -3.0,
                         'digital_max': 32767, 'digital_min': -32768,
                         'prefilter': 'pre1', 'transducer': 'trans1'}

        gender_mapping = {'X': '', 'XX':'', 'XXX':'', '?':'', None:'',
                          'M': 'Male', 'male':'Male', 'man':'Male', 1:'Male',
                          'F':'Female', 'female':'Female', 0:'Female'}

        for gender, expected in gender_mapping.items():
            f = pyedflib.EdfWriter(self.edf_data_file, 2, file_type=pyedflib.FILETYPE_EDFPLUS)
            f.setGender(gender)
            f.setSignalHeader(0, channel_info1)
            f.setSignalHeader(1, channel_info2)
            data = np.ones(100) * 0.1
            assert f.writePhysicalSamples(data)==0, 'error while writing physical sample'
            assert f.writePhysicalSamples(data)==0, 'error while writing physical sample'
            del f
    
            f = pyedflib.EdfReader(self.edf_data_file)
            np.testing.assert_equal(f.getLabel(0), 'test_label1')
            np.testing.assert_equal(f.getPhysicalDimension(0), 'mV')
            np.testing.assert_equal(f.getSampleFrequency(0), 100)
            self.assertEqual(f.getGender(), expected,
                             'set {}, but {}!={} '.format(gender, expected, f.getGender()))
            del f

    def test_non_one_second_record_duration(self):
        channel_count = 4
        record_duration_seconds = 2
        record_duration = record_duration_seconds * 1000 * 100
        sample_frequency = 256
        record_count = 10
        sample_count_per_channel = sample_frequency * record_count

        f = pyedflib.EdfWriter(self.edf_data_file, channel_count, file_type=pyedflib.FILETYPE_EDF)
        f.setDatarecordDuration(record_duration)

        physMax = 1
        physMin = -physMax
        digMax = 32767
        digMin = -digMax

        f.setSignalHeaders([{
            'label': 'test_label{}'.format(idx),
            'sample_frequency': sample_frequency,
            'dimension': 'mV',
            'physical_min': physMin,
            'physical_max': physMax,
            'digital_min': digMin,
            'digital_max': digMax,
            'transducer': 'trans{}'.format(idx),
            'prefilter': 'pre{}'.format(idx)
        } for idx in range(channel_count)])

        f.writeSamples(np.random.rand(channel_count, sample_count_per_channel))
        del f

        f = pyedflib.EdfReader(self.edf_data_file)

        for signal_header in f.getSignalHeaders():
            self.assertEqual(signal_header['sample_frequency'], sample_frequency)

        self.assertEqual(f.datarecord_duration, record_duration_seconds)
        self.assertEqual(f.datarecords_in_file, record_count)

        del f

    def test_sample_rate_backwards_compatibility(self):
        channel_count = 4
        record_duration_seconds = 1
        record_duration = record_duration_seconds * 1000 * 100
        # Choosing a weird sample rate to make sure it doesn't equal any defaults
        sample_rate = 42
        record_count = 10
        sample_count_per_channel = sample_rate * record_count


        physMax = 1
        physMin = -physMax
        digMax = 32767
        digMin = -digMax

        base_signal_header = lambda idx: {
            'label': 'test_label{}'.format(idx),
            'dimension': 'mV',
            'physical_min': physMin,
            'physical_max': physMax,
            'digital_min': digMin,
            'digital_max': digMax,
            'transducer': 'trans{}'.format(idx),
            'prefilter': 'pre{}'.format(idx)
        }

        with self.subTest("when 'sample_frequency' param is missing"):
            f = pyedflib.EdfWriter(self.edf_data_file, channel_count, file_type=pyedflib.FILETYPE_EDF)
            f.setDatarecordDuration(record_duration)

            f.setSignalHeaders([{
                'sample_rate': sample_rate,
                **base_signal_header(idx)
            } for idx in range(channel_count)])

            with self.assertWarnsRegex(DeprecationWarning, "'sample_rate' parameter is deprecated"):
                f.writeSamples(np.random.rand(channel_count, sample_count_per_channel))
                del f

            f = pyedflib.EdfReader(self.edf_data_file)

            for signal_header in f.getSignalHeaders():
                self.assertEqual(signal_header['sample_rate'], sample_rate)

            del f

        with self.subTest("when 'sample_frequency' param is present"):
            f = pyedflib.EdfWriter(self.edf_data_file, channel_count, file_type=pyedflib.FILETYPE_EDF)
            f.setDatarecordDuration(record_duration)
            f.setSignalHeaders([{
                'sample_frequency': sample_rate,
                **base_signal_header(idx)
            } for idx in range(channel_count)])
            f.writeSamples(np.random.rand(channel_count, sample_count_per_channel))

            del f

            f = pyedflib.EdfReader(self.edf_data_file)
            for signal_header in f.getSignalHeaders():
                self.assertEqual(signal_header['sample_frequency'], sample_rate)

            del f


    def test_EdfWriter_more_than_80_chars(self):

        channel_info1 = {'label': 'label',
                         'dimension': 'dim',
                         'sample_frequency': 100,
                         'physical_max': 1,
                         'physical_min': -1.0,
                         'digital_max': 32767,
                         'digital_min': -32768,
                         'prefilter': 'pre',
                         'transducer': 'trans'}

        header = {'birthdate': '',
                  'startdate': datetime(2021, 6, 26, 13, 16, 1),
                  'gender': '',
                  'admincode': '',
                  'equipment': '',
                  'patientcode': 'x'*40,
                  'patient_additional': 'x'*30,
                  'patientname': '',
                  'recording_additional': '',
                  'technician': ''}

        # now 4 warnings should appear.
        with self.assertWarns(UserWarning):
            with pyedflib.EdfWriter(self.edf_data_file, 1, file_type=pyedflib.FILETYPE_EDFPLUS) as f:
                f.setHeader(header)
                f.setSignalHeader(0, channel_info1)
                data = np.ones(100) * 0.1
                f.writePhysicalSamples(data)

        header = {'birthdate': '',
                  'startdate': datetime(2021, 6, 26, 13, 16, 1),
                  'gender': '',
                  'admincode': '',
                  'equipment': 'e'*20,
                  'patientcode': 'x',
                  'patient_additional': 'x',
                  'patientname': '',
                  'recording_additional': 'r'*20,
                  'technician': 't'*20}

        # now 4 warnings should appear.
        with self.assertWarns(UserWarning):
            with pyedflib.EdfWriter(self.edf_data_file, 1, file_type=pyedflib.FILETYPE_EDFPLUS) as f:
                f.setHeader(header)
                f.setSignalHeader(0, channel_info1)
                data = np.ones(100) * 0.1
                f.writePhysicalSamples(data)

        with pyedflib.EdfReader(self.edf_data_file) as f:
            np.testing.assert_equal(f.getEquipment(), 'e'*20)
            np.testing.assert_equal(f.getTechnician(), 't'*20)


    def test_EdfWriter_too_long_headers(self):
        channel_info1 = {'label': 'l'*100, # this should be too long
                         'dimension': 'd'*100,
                         'sample_frequency': 100,
                         'physical_max': 212345.523,
                         'physical_min': -1.0,
                         'digital_max': 32767,
                         'digital_min': -32768,
                         'prefilter': 'p'*100,
                         'transducer': 't'*100}

        # now 4 warnings should appear.
        with self.assertWarns(UserWarning):
            with pyedflib.EdfWriter(self.edf_data_file, 1, file_type=pyedflib.FILETYPE_EDF) as f:
                f.setSignalHeader(0,channel_info1)
                data = np.ones(100) * 0.1
                f.writePhysicalSamples(data)

        with pyedflib.EdfReader(self.edf_data_file) as f:
            np.testing.assert_equal(f.getLabel(0), 'l'*16)
            np.testing.assert_equal(f.getPhysicalDimension(0), 'd'*8)
            np.testing.assert_equal(f.getPrefilter(0), 'p'*80)
            np.testing.assert_equal(f.getTransducer(0), 't'*80)
            np.testing.assert_equal(f.getSampleFrequency(0), 100)
            self.assertEqual(f.filetype, pyedflib.FILETYPE_EDF)


    def test_EdfWriter_out_of_bounds_pmin_pmax_dmin_dmax(self):
        channel_info = {'label': 'l', # this should be too long
                         'dimension': 'd',
                         'sample_frequency': 100,
                         'physical_max': 0.9999999999,
                         'physical_min': -0.9999999999,
                         'digital_max': 32767,
                         'digital_min': -32768,
                         'prefilter': 'p',
                         'transducer': 't'}

        # now a warning should appear.
        with self.assertWarns(UserWarning):
            with pyedflib.EdfWriter(self.edf_data_file, 1, file_type=pyedflib.FILETYPE_EDF) as f:
                f.setSignalHeader(0,channel_info)
                data = np.ones(100) * 0.1
                f.writePhysicalSamples(data)

        with pyedflib.EdfReader(self.edf_data_file) as f:
            self.assertEqual(f.filetype, pyedflib.FILETYPE_EDF)

        with self.assertRaises(ValueError):
            channel_info1 = channel_info.copy()
            channel_info1['physical_max'] = 999999999999
            # now a warning should appear.
            with pyedflib.EdfWriter(self.edf_data_file, 1, file_type=pyedflib.FILETYPE_EDF) as f:
                f.setSignalHeader(0,channel_info1)
                data = np.ones(100) * 0.1
                f.writePhysicalSamples(data)

        with self.assertRaises(ValueError):
            channel_info1 = channel_info.copy()
            channel_info1['physical_min'] = -999999999999
            # now a warning should appear.
            with pyedflib.EdfWriter(self.edf_data_file, 1, file_type=pyedflib.FILETYPE_EDF) as f:
                f.setSignalHeader(0,channel_info1)
                data = np.ones(100) * 0.1
                f.writePhysicalSamples(data)

        # test with DMIN/DMAX for EDF+
        with self.assertRaises(ValueError):
            channel_info1 = channel_info.copy()
            channel_info1['digital_min'] = -32769
            # now a warning should appear.
            with pyedflib.EdfWriter(self.edf_data_file, 1, file_type=pyedflib.FILETYPE_EDF) as f:
                f.setSignalHeader(0,channel_info1)
                data = np.ones(100) * 0.1
                f.writePhysicalSamples(data)

        with self.assertRaises(ValueError):
            channel_info1 = channel_info.copy()
            channel_info1['digital_max'] = 32768
            # now a warning should appear.
            with pyedflib.EdfWriter(self.edf_data_file, 1, file_type=pyedflib.FILETYPE_EDF) as f:
                f.setSignalHeader(0,channel_info1)
                data = np.ones(100) * 0.1
                f.writePhysicalSamples(data)

        # test with DMIN/DMAX for BDF+
        with self.assertRaises(ValueError):
            channel_info1 = channel_info.copy()
            channel_info1['digital_min'] = -8388609
            # now a warning should appear.
            with pyedflib.EdfWriter(self.edf_data_file, 1, file_type=pyedflib.FILETYPE_BDF) as f:
                f.setSignalHeader(0,channel_info1)
                data = np.ones(100) * 0.1
                f.writePhysicalSamples(data)

        with self.assertRaises(ValueError):
            channel_info1 = channel_info.copy()
            channel_info1['digital_max'] = 8388608
            # now a warning should appear.
            with pyedflib.EdfWriter(self.edf_data_file, 1, file_type=pyedflib.FILETYPE_BDF) as f:
                f.setSignalHeader(0,channel_info1)
                data = np.ones(100) * 0.1
                f.writePhysicalSamples(data)

if __name__ == '__main__':
    # run_module_suite(argv=sys.argv)
    unittest.main()
