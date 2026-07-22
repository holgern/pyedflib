# Copyright (c) 2019 - 2023 Simon Kern
# Copyright (c) 2015 Holger Nahrstaedt

import gc
import os

# from numpy.testing import (assert_raises, run_module_suite,
#                            assert_equal, assert_allclose, assert_almost_equal)
import unittest
import warnings
from datetime import date, datetime

import numpy as np

import pyedflib
from pyedflib.edfreader import EdfReader, _debug_parse_header
from pyedflib.edfwriter import (
    ChannelDoesNotExist,
    EdfWriter,
    _calculate_record_duration,
)


class TestEdfWriter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        cls.bdfplus_data_file = os.path.join(data_dir, 'tmp_test_file_plus.bdf')
        cls.edfplus_data_file = os.path.join(data_dir, 'tmp_test_file_plus.edf')
        cls.bdf_data_file = os.path.join(data_dir, 'tmp_test_file.bdf')
        cls.edf_data_file = os.path.join(data_dir, 'tmp_test_file.edf')
        cls.data_dir = data_dir

        cls.ch_info_edf = {'label': 'test_label', 'dimension': 'mV',
                           'sample_frequency': 100,  'physical_max': 1.0,
                           'physical_min': -1.0, 'digital_max': 32767,
                           'digital_min': -32768,  'prefilter': 'pre1',
                           'transducer': 'trans1'}

        cls.ch_info_bdf =  cls.ch_info_edf.copy()
        cls.ch_info_bdf.update({'digital_max': 8388607,
                                'digital_min': -8388608})

    @classmethod
    def tearDownClass(cls):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        tmpfiles = [f for f in os.listdir(data_dir) if f.startswith('tmp')]
        for file in tmpfiles:
            try:
                os.remove(os.path.join(data_dir, file))
            except Exception as e:
                print(e)

    def tearDown(self):
        # small hack to close handles in case of tests throwing an exception
        for obj in gc.get_objects():
            if isinstance(obj, (EdfWriter, EdfReader)):
                obj.close()
                del obj

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
            filename = os.path.join(self.data_dir, f'tmp_write_{file_type}.edf')

            with pyedflib.EdfWriter(filename, 2,
                                file_type=file_type) as f:
                f.setSignalHeader(0, channel_info1)
                f.setSignalHeader(1, channel_info2)
                data = np.random.randint(-32768, 32767, 100)


                for i in range(2):
                    res = f.writePhysicalSamples(data.astype(float))
                    if res<0:
                        print(res, f'Error for filetype {file_type} on writePhysicalSamples signal {i}')
                        error = True
                for i in range(2):
                    res = f.writeDigitalSamples(data.astype(np.int32))
                    if res<0:
                        print(res, f'Error for filetype {file_type} on writeDigitalSamples signal {i}')
                        error = True

                res = f.blockWritePhysicalSamples(np.hstack([data.astype(float)]*2))
                if res<0:
                    print(res, f'Error for filetype {file_type} on blockWritePhysicalSamples signal {i}')
                    error = True

                res = f.blockWriteDigitalSamples(np.hstack([data.astype(np.int32)]*2))
                if res<0:
                    print(res, f'Error for filetype {file_type} on blockWriteDigitalSamples signal {i}')
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
            raise OSError('Writetests not successfully, see log for details')


    def test_subsecond_starttime(self):

        f = pyedflib.EdfWriter(self.edfplus_data_file, 1,
                                file_type=pyedflib.FILETYPE_EDFPLUS)

        startdate = datetime(2017, 1, 2, 13, 14, 15, 250)
        header = {'technician': 'tec1', 'recording_additional': 'recAdd1', 'patientname': 'pat1',
                  'patient_additional': 'patAdd1', 'patientcode': 'code1', 'equipment': 'eq1',
                  'admincode':'admin1','sex':1,'startdate':startdate,'birthdate':date(1951, 8, 2)}
        f.setHeader(header)
        f.setStartdatetime(startdate)
        f.setSignalHeader(0, self.ch_info_edf)
        data = np.ones(100) * 0.1
        assert f.writePhysicalSamples(data)==0, 'error while writing physical sample'
        assert f.writePhysicalSamples(data)==0, 'error while writing physical sample'
        del f

        f = pyedflib.EdfReader(self.edfplus_data_file)
        startdate2 = f.getStartdatetime()
        assert startdate2==startdate, f'write {startdate} != read {startdate2}'
        del f


    def test_annotation_overflow_warning(self):
        # gh-187: annotations exceeding the capacity of
        # n_datarecords * n_annotation_signals are silently dropped by the
        # C library, the writer should warn about this on close()
        def write_file(n_annotations, n_annotation_signals=1):
            f = pyedflib.EdfWriter(self.edfplus_data_file, 1,
                                   file_type=pyedflib.FILETYPE_EDFPLUS)
            f.setSignalHeader(0, self.ch_info_edf)
            if n_annotation_signals > 1:
                f.set_number_of_annotation_signals(n_annotation_signals)
            for _ in range(5):  # 5 datarecords of 1 second
                f.writePhysicalSamples(np.ones(100) * 0.1)
            for i in range(n_annotations):
                f.writeAnnotation(i * 0.1, -1, f'annot_{i}')
            f.close()

        # more annotations than datarecords -> warning
        with self.assertWarns(UserWarning) as cm:
            write_file(n_annotations=8)
        self.assertIn('3 annotation(s) will be lost', str(cm.warning))

        # enough annotation signals -> no warning
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            write_file(n_annotations=8, n_annotation_signals=2)
        self.assertEqual([str(w.message) for w in caught], [])

        # and the file with 2 annotation signals must contain all annotations
        with pyedflib.EdfReader(self.edfplus_data_file) as f:
            self.assertEqual(f.annotations_in_file, 8)

    def test_set_number_of_annotation_signals(self):
        # gh-187: manually increasing the number of annotation signals must
        # actually increase the annotation capacity of the file
        n_records = 5           # 5 datarecords of 1 second
        n_annotation_signals = 10
        n_annotations = 45      # > 5 records, <= 5 records * 10 signals

        f = pyedflib.EdfWriter(self.edfplus_data_file, 1,
                               file_type=pyedflib.FILETYPE_EDFPLUS)
        f.setSignalHeader(0, self.ch_info_edf)
        f.set_number_of_annotation_signals(n_annotation_signals)
        for _ in range(n_records):
            f.writePhysicalSamples(np.ones(100) * 0.1)
        for i in range(n_annotations):
            f.writeAnnotation(round(i * 0.1, 1), 0.5, f'annot_{i}')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            f.close()
        self.assertEqual([str(w.message) for w in caught], [])

        # the file header must contain one label per annotation signal
        with open(self.edfplus_data_file, 'rb') as fh:
            header = fh.read(256 * (1 + 1 + n_annotation_signals))
        self.assertEqual(header.count(b'EDF Annotations'), n_annotation_signals)

        # all annotations must be read back with correct onset/duration/text
        with pyedflib.EdfReader(self.edfplus_data_file) as f:
            self.assertEqual(f.annotations_in_file, n_annotations)
            onsets, durations, texts = f.readAnnotations()
        order = np.argsort(onsets)
        np.testing.assert_allclose(onsets[order],
                                   [round(i * 0.1, 1) for i in range(n_annotations)])
        np.testing.assert_allclose(durations.astype(float)[order], [0.5] * n_annotations)
        self.assertEqual([texts[i] for i in order],
                         [f'annot_{i}' for i in range(n_annotations)])

    def test_subsecond_annotation(self):
        f = pyedflib.EdfWriter(self.bdfplus_data_file, 1,
                               file_type=pyedflib.FILETYPE_BDFPLUS)
        f.setSignalHeader(0, self.ch_info_bdf)
        data = np.ones(100) * 0.1
        f.writePhysicalSamples(data)
        f.writePhysicalSamples(data)
        f.writePhysicalSamples(data)
        f.writePhysicalSamples(data)
        f.writeAnnotation(1.23456, 0.2222, "annotation1_ä")
        f.writeAnnotation(0.2567, -1, "annotation2_ü")
        f.writeAnnotation(1.2567, 0, "annotation3_ö")
        f.writeAnnotation(1.3067, -1, "annotation4_ß")
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
        f = pyedflib.EdfWriter(self.bdfplus_data_file, 2,
                                file_type=pyedflib.FILETYPE_BDFPLUS)
        f.setSignalHeader(0, self.ch_info_bdf)
        f.setSignalHeader(1, self.ch_info_bdf)
        f.setTechnician('tec1')
        f.setRecordingAdditional('recAdd1')
        f.setPatientName('pat1')
        f.setPatientCode('code1')
        f.setPatientAdditional('patAdd1')
        f.setAdmincode('admin1')
        f.setEquipment('eq1')
        f.setSex(1)
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
        np.testing.assert_equal(f.getSex(), 'Male')
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
        f = pyedflib.EdfWriter(self.bdfplus_data_file, 2,
                                file_type=pyedflib.FILETYPE_BDFPLUS)
        f.setSignalHeader(0, self.ch_info_bdf)
        f.setSignalHeader(1, self.ch_info_bdf)
        f.setTechnician('tec1')
        f.setRecordingAdditional('recAdd1')
        f.setPatientName('empty')

        f.setPatientCode('code1')
        f.setPatientAdditional('patAdd1')
        f.setAdmincode('admin1')
        f.setEquipment('eq1')
        f.setSex("Male")
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
        np.testing.assert_equal(f.getSex(), 'Male')
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
        f = pyedflib.EdfWriter(self.bdf_data_file, 2,
                                file_type=pyedflib.FILETYPE_BDF)
        f.setSignalHeader(0, self.ch_info_bdf)
        f.setSignalHeader(1, self.ch_info_bdf)

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

        f = pyedflib.EdfWriter(self.edfplus_data_file, 1,
                                file_type=pyedflib.FILETYPE_EDFPLUS)

        header = {'technician': 'tec1', 'recording_additional': 'recAdd1', 'patientname': 'pat1',
                  'patient_additional': 'patAdd1', 'patientcode': 'code1', 'equipment': 'eq1',
                  'admincode':'admin1','sex':1,'startdate':datetime(2017, 1, 1, 1, 1, 1),'birthdate':date(1951, 8, 2)}
        f.setHeader(header)
        f.setSignalHeader(0, self.ch_info_edf)
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
        np.testing.assert_equal(f.getSex(), 'Male')
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
        f = pyedflib.EdfWriter(self.edf_data_file, 2,
                                file_type=pyedflib.FILETYPE_EDF)
        f.setSignalHeader(0, self.ch_info_edf)
        f.setSignalHeader(1, self.ch_info_edf)
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
        f = pyedflib.EdfWriter(self.bdfplus_data_file, 2,
                              file_type=pyedflib.FILETYPE_BDFPLUS)
        f.setSignalHeader(0, self.ch_info_bdf)
        f.setSignalHeader(1, self.ch_info_bdf)

        data1 = np.ones(500) * 0.1
        data2 = np.ones(500) * 0.2
        data_list = [data1, data2]
        f.writeSamples(data_list)
        f.close()

        f = pyedflib.EdfReader(self.bdfplus_data_file)
        data1_read = f.readSignal(0)
        data2_read = f.readSignal(1)
        f._close()
        np.testing.assert_equal(len(data1), len(data1_read))
        np.testing.assert_equal(len(data2), len(data2_read))
        np.testing.assert_almost_equal(data1, data1_read)
        np.testing.assert_almost_equal(data2, data2_read)
        self.assertEqual(f.filetype, pyedflib.FILETYPE_BDFPLUS)

    def test_EdfWriter_EDF_contextmanager(self):

        with pyedflib.EdfWriter(self.edf_data_file, 2, file_type=pyedflib.FILETYPE_EDF) as f:
            f.setSignalHeader(0, self.ch_info_edf)
            f.setSignalHeader(1, self.ch_info_edf)
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
        with pyedflib.EdfWriter(self.bdfplus_data_file, 2, file_type=pyedflib.FILETYPE_BDFPLUS) as f:

            f.setSignalHeader(0, self.ch_info_bdf)
            f.setSignalHeader(1, self.ch_info_bdf)
            data1 = np.ones(500) * 0.1
            data2 = np.ones(500) * 0.2
            data_list = [data1, data2]
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
        f = pyedflib.EdfWriter(self.bdfplus_data_file, 2,
                              file_type=pyedflib.FILETYPE_BDFPLUS)
        f.setSignalHeader(0, self.ch_info_bdf)
        f.setSignalHeader(1, self.ch_info_bdf)

        data1 = np.ones(500) * 0.1
        data2 = np.ones(500) * 0.2
        data_list = [data1, data2]
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
        data_list = [data1, data2]
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

        f = pyedflib.EdfWriter(self.edfplus_data_file, 1,
                              file_type=pyedflib.FILETYPE_EDFPLUS)
        f.setSignalHeader(0, self.ch_info_edf)

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
        f.setSignalHeader(0, self.ch_info_edf)

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
        f = pyedflib.EdfWriter(self.bdfplus_data_file, 1,
                               file_type=pyedflib.FILETYPE_BDFPLUS)
        f.setSignalHeader(0, self.ch_info_bdf)
        data = np.ones(100) * 0.1
        f.writePhysicalSamples(data)
        f.writePhysicalSamples(data)
        f.writePhysicalSamples(data)
        f.writePhysicalSamples(data)
        f.writeAnnotation(1.23, 0.2, "annotation1_ä")
        f.writeAnnotation(0.25, -1, "annotation2_ü")
        f.writeAnnotation(1.25, 0, "annotation3_ö")
        f.writeAnnotation(1.30, -1, "annotation4_ß")
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

    def test_AnnotationWriting_latin(self):
        """test that non-ASCII chars are simply omitted"""
        f = pyedflib.EdfWriter(self.bdfplus_data_file, 1,
                                file_type=pyedflib.FILETYPE_BDFPLUS)
        f.setSignalHeader(0, self.ch_info_bdf)
        data = np.ones(100) * 0.1
        f.writePhysicalSamples(data)
        f.writePhysicalSamples(data)
        f.writePhysicalSamples(data)
        f.writePhysicalSamples(data)
        f.writeAnnotation(1.23, 0.2, "Zähne")
        f.writeAnnotation(0.25, -1, "Fuß")
        f.writeAnnotation(1.25, 0, "abc")
        del f

        f = pyedflib.EdfReader(self.bdfplus_data_file)
        self.assertEqual(f.filetype, pyedflib.FILETYPE_BDFPLUS)

        ann_time, ann_duration, ann_text = f.readAnnotations()
        del f
        np.testing.assert_almost_equal(ann_time[0], 1.23)
        np.testing.assert_almost_equal(ann_duration[0], 0.2)
        np.testing.assert_equal(ann_text[0], "Zähne")
        np.testing.assert_almost_equal(ann_time[1], 0.25)
        np.testing.assert_almost_equal(ann_duration[1], -1)
        np.testing.assert_equal(ann_text[1], "Fuß")
        np.testing.assert_almost_equal(ann_time[2], 1.25)
        np.testing.assert_almost_equal(ann_duration[2], 0)
        np.testing.assert_equal(ann_text[2], "abc")


    def test_BytesChars(self):
        f = pyedflib.EdfWriter(self.bdfplus_data_file, 1,
                                file_type=pyedflib.FILETYPE_BDFPLUS)
        f.setSignalHeader(0, self.ch_info_bdf)
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


    def test_sex_setting_correctly(self):
        channel_info1 = {'label': 'test_label1', 'dimension': 'mV', 'sample_frequency': 100,
                         'physical_max': 3.0, 'physical_min': -3.0,
                         'digital_max': 32767, 'digital_min': -32768,
                         'prefilter': 'pre1', 'transducer': 'trans1'}
        channel_info2 = {'label': 'test_label2', 'dimension': 'mV', 'sample_frequency': 100,
                         'physical_max': 3.0, 'physical_min': -3.0,
                         'digital_max': 32767, 'digital_min': -32768,
                         'prefilter': 'pre1', 'transducer': 'trans1'}

        sex_mapping = {'X': '', 'XX':'', 'XXX':'', '?':'', None:'',
                          'M': 'Male', 'male':'Male', 'man':'Male', 1:'Male',
                          'F':'Female', 'female':'Female', 0:'Female'}

        for sex, expected in sex_mapping.items():
            f = pyedflib.EdfWriter(self.edf_data_file, 2, file_type=pyedflib.FILETYPE_EDFPLUS)
            f.setSex(sex)
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
            self.assertEqual(f.getSex(), expected,
                             f'set {sex}, but f.getSex()!={expected}')
            with self.assertWarns(DeprecationWarning):
                self.assertEqual(f.getGender(), expected,
                             f'set {sex}, but f.getGender()!={expected}')
            del f

        # try again, this time with setGender() instead of setSex()
        for sex, expected in sex_mapping.items():
            f = pyedflib.EdfWriter(self.edf_data_file, 2, file_type=pyedflib.FILETYPE_EDFPLUS)
            f.setGender(sex)  # deprecated
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
            self.assertEqual(f.getSex(), expected,
                             f'set {sex}, but f.getSex()!={expected}')
            with self.assertWarns(DeprecationWarning):
                self.assertEqual(f.getGender(), expected,
                             f'set {sex}, but f.getGender()!={expected}')
            del f

    def test_non_one_second_record_duration(self):
        channel_count = 4
        record_duration = 2
        samples_per_record = 256
        sample_frequency = samples_per_record/record_duration
        record_count = 4

        f = pyedflib.EdfWriter(self.edf_data_file, channel_count, file_type=pyedflib.FILETYPE_EDF)
        f.setDatarecordDuration(record_duration)

        physMax = 1
        physMin = -physMax
        digMax = 32767
        digMin = -digMax

        f.setSignalHeaders([{
            'label': f'test_label{idx}',
            'sample_frequency': sample_frequency,
            'dimension': 'mV',
            'physical_min': physMin,
            'physical_max': physMax,
            'digital_min': digMin,
            'digital_max': digMax,
            'transducer': f'trans{idx}',
            'prefilter': f'pre{idx}'
        } for idx in range(channel_count)])

        f.writeSamples(np.random.rand(channel_count, samples_per_record*4))
        f.close()
        del f

        f = pyedflib.EdfReader(self.edf_data_file)

        for signal_header in f.getSignalHeaders():
            self.assertEqual(signal_header['sample_frequency'], sample_frequency)

        self.assertEqual(f.datarecord_duration, record_duration)
        self.assertEqual(f.datarecords_in_file, record_count)
        f.close()
        del f


    def test_force_record_duration(self):
        """forcing a specific record duration should alter the sample_freq"""
        channel_count = 4
        record_duration = 0.33333  # should not be able to represent 256 Hz
        samples_per_record = 256
        sample_frequency = 256
        sample_freq_exp = int(256*record_duration)/record_duration

        f = pyedflib.EdfWriter(self.edf_data_file, channel_count, file_type=pyedflib.FILETYPE_EDF)
        with self.assertWarns(UserWarning):
            f.setDatarecordDuration(record_duration)

        physMax = 1
        physMin = -physMax
        digMax = 32767
        digMin = -digMax

        f.setSignalHeaders([{
            'label': f'test_label{idx}',
            'sample_frequency': sample_frequency,
            'dimension': 'mV',
            'physical_min': physMin,
            'physical_max': physMax,
            'digital_min': digMin,
            'digital_max': digMax,
            'transducer': f'trans{idx}',
            'prefilter': f'pre{idx}'
        } for idx in range(channel_count)])

        f.writeSamples(np.random.rand(channel_count, samples_per_record*4))
        f.close()
        del f

        f = pyedflib.EdfReader(self.edf_data_file)

        for signal_header in f.getSignalHeaders():
            self.assertEqual(signal_header['sample_frequency'], sample_freq_exp)

        self.assertEqual(f.datarecord_duration, record_duration)
        f.close()
        del f


    def test_sample_rate_deprecation(self):
        channel_count = 4
        record_duration = 1
        # Choosing a weird sample rate to make sure it doesn't equal any defaults
        sample_rate = 42

        physMax = 1
        physMin = -physMax
        digMax = 32767
        digMin = -digMax

        def base_signal_header(idx):
            return {
                    'label': f'test_label{idx}',
                    'dimension': 'mV',
                    'physical_min': physMin,
                    'physical_max': physMax,
                    'digital_min': digMin,
                    'digital_max': digMax,
                    'transducer': f'trans{idx}',
                    'prefilter': f'pre{idx}'
                }

        f = pyedflib.EdfWriter(self.edf_data_file, channel_count, file_type=pyedflib.FILETYPE_EDF)
        f.setDatarecordDuration(record_duration)
        with self.assertRaises(FutureWarning):
            f.setSignalHeaders([{
                'sample_rate': sample_rate,
                **base_signal_header(idx)
            } for idx in range(channel_count)])
        del f



    def test_EdfWriter_more_than_80_chars(self):

        header = {'birthdate': '',
                  'startdate': datetime(2021, 6, 26, 13, 16, 1),
                  'sex': '',
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
                f.setSignalHeader(0,  self.ch_info_edf)
                data = np.ones(100) * 0.1
                f.writePhysicalSamples(data)

        header = {'birthdate': '',
                  'startdate': datetime(2021, 6, 26, 13, 16, 1),
                  'sex': '',
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
                f.setSignalHeader(0, self.ch_info_edf)
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


    def test_EdfWriter_float_sample_frequency(self):

        # create 4 channels with mixed sample frequencies
        sfreqs = [256, 10, 5.5, 0.1, 19.8, 10000]
        channel_info = [{'sample_frequency': fs} for fs in sfreqs]
        f = pyedflib.EdfWriter(self.edfplus_data_file, len(sfreqs),
                                file_type=pyedflib.FILETYPE_EDFPLUS)

        f.setSignalHeaders(channel_info)
        data = [np.random.randint(-100, 100, int(fs*30))/100 for fs in sfreqs]

        f.writeSamples(data)
        f.close()
        del f

        # read back data with mixed sfreq and check all data is correct
        f = pyedflib.EdfReader(self.edfplus_data_file)
        for i, (sig, fs) in enumerate(zip(data, sfreqs)):
            d = f.readSignal(i)
            np.testing.assert_almost_equal(sig, d, decimal=4)
            np.testing.assert_equal(f.getSampleFrequency(i), fs)

        f.close()
        del f


    def test_write_annotations_long_long(self):
        """check that very long recordings can store annotations"""
        # 2 channels for one week, write annotation every two hours
        fs = 5
        data = np.random.normal(size=(2,7*24*60*60*fs))
        ch_names = ['EEG1', 'EEG2']

        with EdfWriter(self.edf_data_file, len(ch_names)) as f:
            f.setSignalHeaders([{'label': 'EEG1',
              'dimension': 'uV',
              'sample_frequency': 256,
              'physical_min': -200.0,
              'physical_max': 200.0,
              'digital_min': -32768,
              'digital_max': 32767,
              'transducer': '',
              'prefilter': ''},
             {'label': 'EEG2',
              'dimension': 'uV',
              'sample_frequency': 256,
              'physical_min': -200.0,
              'physical_max': 200.0,
              'digital_min': -32768,
              'digital_max': 32767,
              'transducer': '',
              'prefilter': ''}])

            f.writeSamples(data)
            for h in range(0, 4*24, 2):
                f.writeAnnotation(h*3600, -1, f"{h} hour after start")

        with pyedflib.EdfReader(self.edf_data_file) as f:
            annotations = f.readAnnotations()
            self.assertEqual( len(annotations[0]), 48)
            np.testing.assert_array_equal(annotations[0], np.arange(0, 342000, 3600*2))

    def test_find_record_duration(self):
        """check that _calculate_record_duration finds optimal values"""
        freqs = np.arange(10)
        duration = _calculate_record_duration(freqs)
        assert duration==1

        freqs = np.arange(10)/3
        duration = _calculate_record_duration(freqs)
        assert duration==3

        freqs = np.array([0.32, 1.4, 2.3, 6.4, 3.1])
        duration = _calculate_record_duration(freqs)
        assert duration == 50

        # float rounding errors should be handled correctly
        freqs = np.array([199.99999999, 255.99999999])
        duration = _calculate_record_duration(freqs)
        assert duration == 1

        for _ in range(10):
            freqs = np.random.randint(1, 10000, 25,).astype(int)
            duration = _calculate_record_duration(freqs)
            assert duration == 1

    def test_record_durations(self):
        """use different record durations and look in the raw header if all seems right"""
        for record_duration in [0.000001, 0.0001, 0.009999, 0.001, 0.01, 0.1, 1, 10, 60]:
            channel_count = 1
            sample_frequency = 1/record_duration

            f = pyedflib.EdfWriter(self.edf_data_file, 1,
                                   file_type=pyedflib.FILETYPE_EDF)
            f.setDatarecordDuration(record_duration)

            digMax = 32767
            digMin = -digMax

            f.setSignalHeaders([{
                'label': 'test_label',
                'sample_frequency': sample_frequency,
                'dimension': 'mV',
                'physical_min': 0,
                'physical_max': 1,
                'digital_min': digMin,
                'digital_max': digMax,
                'transducer': '',
                'prefilter': ''
                }])

            f.writeSamples(np.random.rand(channel_count, 1000))
            f.close()
            del f

            header, _sheader = _debug_parse_header(self.edf_data_file, printout=False)
            self.assertEqual(float(header['record_duration']), record_duration)

            with pyedflib.EdfReader(self.edf_data_file) as f:
                self.assertEqual(f.datarecord_duration, record_duration)
                f.close()


        for record_duration in [0.0000005, 61, 0.0001234567]:
            channel_count = 1
            sample_frequency = 1/record_duration

            with pyedflib.EdfWriter(self.edf_data_file, 1,
                                   file_type=pyedflib.FILETYPE_EDF) as f:
                with self.assertRaises(ValueError):
                    f.setDatarecordDuration(record_duration)


    def test_write_annotations_utf8(self):
        """properly test for UTF8 writing of characters, not just umlauts"""

        f = pyedflib.EdfWriter(self.bdfplus_data_file, 1,
                               file_type=pyedflib.FILETYPE_BDFPLUS)
        f.setSignalHeader(0, self.ch_info_bdf)
        f.writePhysicalSamples(np.ones(100))
        utf8_string = '中文测试八个字'
        f.writeAnnotation(1.23, 0.2, utf8_string)
        del f

        f = pyedflib.EdfReader(self.bdfplus_data_file)
        ann_time, ann_duration, ann_text = f.readAnnotations()
        del f

        np.testing.assert_almost_equal(ann_time[0], 1.23)
        np.testing.assert_almost_equal(ann_duration[0], 0.2)
        np.testing.assert_equal(ann_text[0], utf8_string)

    def _buffered_ch_info(self, fs):
        return dict(self.ch_info_edf, sample_frequency=fs,
                    physical_min=-5, physical_max=5)

    def test_buffered_writer_prevents_inflation(self):
        fs = 1000
        x = np.random.default_rng(0).standard_normal(10 * fs) * 0.1
        filename = os.path.join(self.data_dir, 'tmp_buffered.edf')

        f = pyedflib.EdfWriter(filename, 1, buffered=True)
        f.setSignalHeader(0, self._buffered_ch_info(fs))
        for s in range(0, len(x), 100):     # 100-sample streaming blocks
            f.writeSamples([x[s:s + 100]])
        f.close()

        f = pyedflib.EdfReader(filename)
        duration = f.file_duration
        back = f.readSignal(0)
        f.close()

        self.assertAlmostEqual(duration, 10.0)  # no inflation
        rms_ratio = np.sqrt(np.mean(back ** 2)) / np.sqrt(np.mean(x ** 2))
        self.assertLess(abs(rms_ratio - 1.0), 0.02)  # amplitude preserved

    def test_naive_subrecord_blocks_inflate_file(self):
        """Documents the behaviour that buffered=True avoids."""
        fs = 1000
        x = np.random.default_rng(0).standard_normal(10 * fs) * 0.1
        filename = os.path.join(self.data_dir, 'tmp_naive.edf')

        f = pyedflib.EdfWriter(filename, 1)
        f.setSignalHeader(0, self._buffered_ch_info(fs))
        for s in range(0, len(x), 100):
            f.writeSamples([np.ascontiguousarray(x[s:s + 100])])
        f.close()

        f = pyedflib.EdfReader(filename)
        duration = f.file_duration
        f.close()

        self.assertEqual(duration, 100.0)   # 10x inflation without buffering

    def test_buffered_writer_multichannel(self):
        fs = 1000
        rng = np.random.default_rng(1)
        x = [rng.standard_normal(10 * fs) * 0.1 for _ in range(2)]
        filename = os.path.join(self.data_dir, 'tmp_buffered_mc.edf')

        f = pyedflib.EdfWriter(filename, 2, buffered=True)
        f.setSignalHeaders([self._buffered_ch_info(fs) for _ in range(2)])
        for s in range(0, len(x[0]), 100):
            f.writeSamples([x[0][s:s + 100], x[1][s:s + 100]])
        f.close()

        f = pyedflib.EdfReader(filename)
        self.assertAlmostEqual(f.file_duration, 10.0)
        f.close()

    def test_buffered_writer_pads_final_record(self):
        """An incomplete final record is committed zero-padded on close()."""
        fs = 100
        x = np.ones(int(2.5 * fs))
        filename = os.path.join(self.data_dir, 'tmp_buffered_pad.edf')

        f = pyedflib.EdfWriter(filename, 1, buffered=True)
        f.setSignalHeader(0, self._buffered_ch_info(fs))
        for s in range(0, len(x), 30):
            f.writeSamples([x[s:s + 30]])
        f.close()

        f = pyedflib.EdfReader(filename)
        duration = f.file_duration
        back = f.readSignal(0)
        f.close()

        self.assertEqual(duration, 3.0)     # 2.5s of data -> 3 records
        np.testing.assert_allclose(back[:len(x)], x, atol=1e-3)
        np.testing.assert_allclose(back[len(x):], 0, atol=1e-3)

    def test_buffered_writer_digital(self):
        fs = 100
        x = (np.arange(5 * fs) % 2000).astype(np.int32)
        filename = os.path.join(self.data_dir, 'tmp_buffered_digital.edf')

        f = pyedflib.EdfWriter(filename, 1, buffered=True)
        f.setSignalHeader(0, self._buffered_ch_info(fs))
        for s in range(0, len(x), 30):
            f.writeSamples([x[s:s + 30]], digital=True)
        f.close()

        f = pyedflib.EdfReader(filename)
        self.assertAlmostEqual(f.file_duration, 5.0)
        back = f.readSignal(0, digital=True)
        f.close()
        np.testing.assert_array_equal(back, x)

    def test_buffered_writer_rejects_mixed_digital(self):
        fs = 100
        filename = os.path.join(self.data_dir, 'tmp_buffered_mixed.edf')

        f = pyedflib.EdfWriter(filename, 1, buffered=True)
        f.setSignalHeader(0, self._buffered_ch_info(fs))
        f.writeSamples([np.zeros(30, dtype=np.int32)], digital=True)
        with self.assertRaises(TypeError):
            f.writeSamples([np.zeros(30)], digital=False)
        f.close()

    def test_buffered_writer_full_records_unaffected(self):
        """Passing whole records at once behaves identically to buffered=False."""
        fs = 100
        x = np.random.default_rng(2).standard_normal(5 * fs) * 0.1
        filename = os.path.join(self.data_dir, 'tmp_buffered_full.edf')

        f = pyedflib.EdfWriter(filename, 1, buffered=True)
        f.setSignalHeader(0, self._buffered_ch_info(fs))
        f.writeSamples([x])
        f.close()

        f = pyedflib.EdfReader(filename)
        self.assertAlmostEqual(f.file_duration, 5.0)
        back = f.readSignal(0)
        f.close()
        np.testing.assert_allclose(back, x, atol=1e-3)

    def test_buffered_writer_pad_with_last(self):
        """pad_with='last' repeats the last sample instead of zero-padding."""
        fs = 100
        x = np.full(int(2.5 * fs), 3.0)
        filename = os.path.join(self.data_dir, 'tmp_buffered_pad_last.edf')

        f = pyedflib.EdfWriter(filename, 1, buffered=True, pad_with='last')
        f.setSignalHeader(0, self._buffered_ch_info(fs))
        for s in range(0, len(x), 30):
            f.writeSamples([x[s:s + 30]])
        f.close()

        f = pyedflib.EdfReader(filename)
        duration = f.file_duration
        back = f.readSignal(0)
        f.close()

        self.assertEqual(duration, 3.0)
        np.testing.assert_allclose(back, 3.0, atol=1e-3)  # no discontinuity

    def test_buffered_writer_pad_with_number(self):
        """pad_with accepts a fixed numeric fill value."""
        fs = 100
        x = np.ones(int(2.5 * fs))
        filename = os.path.join(self.data_dir, 'tmp_buffered_pad_number.edf')

        f = pyedflib.EdfWriter(filename, 1, buffered=True, pad_with=-2.5)
        f.setSignalHeader(0, self._buffered_ch_info(fs))
        for s in range(0, len(x), 30):
            f.writeSamples([x[s:s + 30]])
        f.close()

        f = pyedflib.EdfReader(filename)
        back = f.readSignal(0)
        f.close()

        np.testing.assert_allclose(back[:len(x)], x, atol=1e-3)
        np.testing.assert_allclose(back[len(x):], -2.5, atol=1e-3)

    def test_buffered_writer_pad_with_digital(self):
        """pad_with is interpreted as a digital value when digital=True."""
        fs = 100
        x = np.full(int(2.5 * fs), 100, dtype=np.int32)
        filename = os.path.join(self.data_dir, 'tmp_buffered_pad_digital.edf')

        f = pyedflib.EdfWriter(filename, 1, buffered=True, pad_with=7)
        f.setSignalHeader(0, self._buffered_ch_info(fs))
        for s in range(0, len(x), 30):
            f.writeSamples([x[s:s + 30]], digital=True)
        f.close()

        f = pyedflib.EdfReader(filename)
        back = f.readSignal(0, digital=True)
        f.close()

        np.testing.assert_array_equal(back[:len(x)], x)
        np.testing.assert_array_equal(back[len(x):], 7)

    def test_buffered_writer_pad_with_invalid(self):
        filename = os.path.join(self.data_dir, 'tmp_buffered_pad_invalid.edf')
        with self.assertRaises(ValueError):
            pyedflib.EdfWriter(filename, 1, buffered=True, pad_with='bogus')
        # a rejected argument must not leave a half-created file behind
        self.assertFalse(os.path.exists(filename))

    def test_pad_with_applies_without_buffering(self):
        """pad_with also controls padding on the regular (unbuffered) path."""
        fs = 100
        x = np.full(int(2.5 * fs), 3.0)
        filename = os.path.join(self.data_dir, 'tmp_pad_unbuffered.edf')

        f = pyedflib.EdfWriter(filename, 1, pad_with='last')
        f.setSignalHeader(0, self._buffered_ch_info(fs))
        f.writeSamples([x])
        f.close()

        f = pyedflib.EdfReader(filename)
        back = f.readSignal(0)
        f.close()

        np.testing.assert_allclose(back, 3.0, atol=1e-3)  # no step back to 0

    def test_pad_with_last_uses_already_written_sample(self):
        """A channel whose buffer is empty pads with its last written sample.

        With differing sample frequencies one channel can end exactly on a
        record boundary while another still has leftover samples. The padding
        for the former must not fall back to zero.
        """
        filename = os.path.join(self.data_dir, 'tmp_pad_mixed_fs.edf')

        f = pyedflib.EdfWriter(filename, 2, buffered=True, pad_with='last')
        f.setSignalHeaders([self._buffered_ch_info(100),
                            self._buffered_ch_info(10)])
        # ch0 keeps 5 leftover samples, ch1 is consumed exactly
        f.writeSamples([np.full(105, 3.0), np.full(10, 4.0)])
        f.close()

        f = pyedflib.EdfReader(filename)
        ch0, ch1 = f.readSignal(0), f.readSignal(1)
        f.close()

        np.testing.assert_allclose(ch0, 3.0, atol=1e-3)
        np.testing.assert_allclose(ch1, 4.0, atol=1e-3)

    def test_pad_with_accepts_numpy_scalars(self):
        """pad_with=signal[-1] (a numpy scalar) must be accepted."""
        filename = os.path.join(self.data_dir, 'tmp_pad_numpy_scalar.edf')
        for value in (np.float64(1.5), np.float32(1.5), np.int32(2)):
            f = pyedflib.EdfWriter(filename, 1, buffered=True, pad_with=value)
            f.close()
        with self.assertRaises(ValueError):
            pyedflib.EdfWriter(filename, 1, buffered=True, pad_with=True)

    def test_pad_with_digital_requires_int(self):
        """A fractional pad value would be silently truncated when digital."""
        fs = 100
        filename = os.path.join(self.data_dir, 'tmp_pad_digital_float.edf')

        f = pyedflib.EdfWriter(filename, 1, buffered=True, pad_with=2.7)
        f.setSignalHeader(0, self._buffered_ch_info(fs))
        with self.assertRaises(TypeError):
            f.writeSamples([np.full(150, 100, dtype=np.int32)], digital=True)
        f.close()

    def test_pad_with_out_of_range_warns(self):
        fs = 100
        filename = os.path.join(self.data_dir, 'tmp_pad_out_of_range.edf')

        f = pyedflib.EdfWriter(filename, 1, buffered=True, pad_with=1000.0)
        f.setSignalHeader(0, self._buffered_ch_info(fs))
        f.writeSamples([np.full(150, 1.0)])
        with self.assertWarns(UserWarning):
            f.close()

    def test_close_on_partially_constructed_writer(self):
        """__del__ must not raise if __init__ failed before opening the file."""
        filename = os.path.join(self.data_dir, 'tmp_partial.edf')
        with self.assertRaises(ValueError):
            pyedflib.EdfWriter(filename, 1, pad_with='bogus')
        gc.collect()  # triggers __del__ -> close() on the dead object


if __name__ == '__main__':
    # run_module_suite(argv=sys.argv)
    unittest.main()
