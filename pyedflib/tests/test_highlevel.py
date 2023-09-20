# Copyright (c) 2019 - 2020 Simon Kern
# Copyright (c) 2015 Holger Nahrstaedt

import os
import shutil
import gc
import numpy as np
# from numpy.testing import (assert_raises, run_module_suite,
#                            assert_equal, assert_allclose, assert_almost_equal)
import unittest
from pyedflib import highlevel
from pyedflib.edfwriter import EdfWriter
from pyedflib.edfreader import EdfReader
from datetime import datetime, timedelta


def _compare_cropped_edf(path_orig_edf, path_cropped_edf):
    # Load original EDF
    orig_signals, orig_signal_headers, orig_header = highlevel.read_edf(path_orig_edf)  # noqa: E501
    orig_start = orig_header["startdate"]

    # Load cropped EDF
    signals, signal_headers, header = highlevel.read_edf(path_cropped_edf)  # noqa: E501
    start = header["startdate"]
    duration = signals[0].size / signal_headers[0]["sample_frequency"]
    stop = start + timedelta(seconds=duration)

    # Compare signal headers
    assert signal_headers == orig_signal_headers

    # Compare signal values
    for i in range(signals.shape[0]):
        sf_sig = signal_headers[i]["sample_frequency"]
        idx_start = int(np.round((start - orig_start).seconds * sf_sig))
        idx_stop = int(np.round((stop - orig_start).seconds * sf_sig))
        assert (signals[i] == orig_signals[i, idx_start:idx_stop]).all()


class TestHighLevel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        data_dir = os.path.join(os.path.dirname(__file__), 'data')

        cls.edfplus_data_file = os.path.join(data_dir, 'tmp_test_file_plus.edf')
        cls.test_generator = os.path.join(data_dir, 'test_generator.edf')
        cls.test_accented = os.path.join(data_dir, "tmp_áä'üöß.edf")
        cls.test_unicode = os.path.join(data_dir, "tmp_utf8-中文źąşㆆ운ʷᨄⅡəПр🤖.edf")
        cls.anonymized = os.path.join(data_dir, "tmp_anonymized.edf")
        cls.personalized = os.path.join(data_dir, "tmp_personalized.edf")
        cls.drop_from = os.path.join(data_dir, 'tmp_drop_from.edf')
        cls.tmp_testfile = os.path.join(data_dir, 'tmp')

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

        header = highlevel.make_header(technician='tech', recording_additional='r_add',
                                                patientname='name', patient_additional='p_add',
                                                patientcode='42', equipment='eeg', admincode='120',
                                                sex='Male', startdate=startdate,birthdate='05.09.1980')
        annotations = [[0.01, -1, 'begin'],[0.5, -1, 'middle'],[10, -1, 'end']]

        signal_headers1 = highlevel.make_signal_headers(['ch'+str(i) for i in range(5)])

        for file_type in [-1,0,1,2,3]:
            if file_type in [0, 2]:
                header['annotations'] = []
            else:
                header['annotations'] = annotations

            file = f'{self.tmp_testfile}_{file_type}_phys.edf'
            signals = np.random.rand(5, 256*300)*200 #5 minutes of eeg
            success = highlevel.write_edf(file, signals, signal_headers1, header, file_type=file_type)
            self.assertTrue(os.path.isfile(file))
            self.assertGreater(os.path.getsize(file), 0)
            self.assertTrue(success)

            signals2, signal_headers2, header2 = highlevel.read_edf(file)

            self.assertEqual(len(signals2), 5)
            self.assertEqual(len(signals2), len(signal_headers2))
            for shead1, shead2 in zip(signal_headers1, signal_headers2):
                # When only 'sample_rate' is present, we use its value to write
                # the file, ignoring 'sample_frequency', which means that when
                # we read it back only the 'sample_rate' value is present.
                self.assertDictEqual({**shead1, 'sample_frequency': shead1['sample_rate']},
                                     shead2)
            np.testing.assert_allclose(signals, signals2, atol=0.01)
            if file_type in [-1, 1, 3]:
                self.assertDictEqual(header, header2)

            file = f'{self.tmp_testfile}_{file_type}_dig.edf'
            signals = (signals*100).astype(np.int8)
            success = highlevel.write_edf(file, signals,  signal_headers1, header, digital=True)
            self.assertTrue(os.path.isfile(file))
            self.assertGreater(os.path.getsize(file), 0)
            self.assertTrue(success)

            signals2, signal_headers2, header2 = highlevel.read_edf(file, digital=True)

            self.assertEqual(len(signals2), 5)
            self.assertEqual(len(signals2), len(signal_headers2))
            np.testing.assert_array_equal(signals, signals2)
            for shead1, shead2 in zip(signal_headers1, signal_headers2):
                # When only 'sample_rate' is present, we use its value to write
                # the file, ignoring 'sample_frequency', which means that when
                # we read it back only the 'sample_rate' value is present.
                self.assertDictEqual({**shead1, 'sample_frequency': shead1['sample_rate']},
                                     shead2)
            # EDF/BDF header writing does not correctly work yet
            if file_type in [-1, 1, 3]:
                self.assertDictEqual(header, header2)

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
        signals2, _, _ = highlevel.read_edf(self.edfplus_data_file, digital=True, verbose=True)
        np.testing.assert_allclose(signals, signals2)
        signals = np.random.rand(3, 256*60) # then rescale to 0-1
        signals = (signals - signals.min()) / (signals.max() - signals.min())
        highlevel.write_edf_quick(self.edfplus_data_file, signals, sfreq=256)
        signals2, _, _ = highlevel.read_edf(self.edfplus_data_file)
        np.testing.assert_allclose(signals, signals2, atol=0.00002)

    def test_fortran_write(self):
        # Create Fortran contiguous array
        signals = np.random.randint(-2048,2048,[4, 5000000])
        signals = np.asfortranarray(signals)
        # Write
        highlevel.write_edf_quick(self.edfplus_data_file, signals.astype(np.int32), sfreq=250, digital=True)
        # Read and check
        signals2, _, _ = highlevel.read_edf(self.edfplus_data_file, digital=True, verbose=True)
        np.testing.assert_allclose(signals, signals2)

        # Create Fortran contiguous list
        signals = [np.random.randint(-2048,2048,(5000000,), dtype=np.int32)]*4
        # Write
        highlevel.write_edf_quick(self.edfplus_data_file, signals, sfreq=250, digital=True)
        # Read and check
        signals2, _, _ = highlevel.read_edf(self.edfplus_data_file, digital=True, verbose=True)
        np.testing.assert_allclose(signals, signals2)


    def test_read_write_decimal_sample_frequencies(self):
        # first test with digital signals
        signals = np.random.randint(-2048, 2048, [3, 256*60+8])
        highlevel.write_edf_quick(self.edfplus_data_file, signals.astype(np.int32), sfreq=8.5, digital=True)
        signals2, _, _ = highlevel.read_edf(self.edfplus_data_file, digital=True, verbose=True)
        np.testing.assert_allclose(signals, signals2)

        # now with physical signals
        signals = np.random.rand(3, 256*60+8) # then rescale to 0-1
        signals = (signals - signals.min()) / (signals.max() - signals.min())
        highlevel.write_edf_quick(self.edfplus_data_file, signals, sfreq=8.5, digital=False)
        signals2, _, _ = highlevel.read_edf(self.edfplus_data_file, digital=False, verbose=True)
        np.testing.assert_allclose(signals, signals2, atol=0.0001)


    def test_read_write_diff_sfreq(self):

        signals = []
        sfreqs = [1, 64, 128, 200]
        sheaders = []
        for sfreq in sfreqs:
            signals.append(np.random.randint(-2048, 2048, sfreq*60).astype(np.int32))
            shead = highlevel.make_signal_header(f'ch{sfreq}', sample_frequency=sfreq)
            sheaders.append(shead)
        highlevel.write_edf(self.edfplus_data_file, signals, sheaders, digital=True)
        signals2, sheaders2, _ = highlevel.read_edf(self.edfplus_data_file, digital=True)
        for s1, s2 in zip(signals, signals2):
            np.testing.assert_allclose(s1, s2)

    def test_assertion_dmindmax(self):

        # test digital and dmin wrong
        signals =[np.random.randint(-2048, 2048, 256*60).astype(np.int32)]
        sheaders = [highlevel.make_signal_header('ch1', sample_frequency=256)]
        sheaders[0]['digital_min'] = -128
        sheaders[0]['digital_max'] = 128
        with self.assertRaises(AssertionError):
            highlevel.write_edf(self.edfplus_data_file, signals, sheaders, digital=True)

        # test pmin wrong
        signals = [np.random.randint(-2048, 2048, 256*60)]
        sheaders = [highlevel.make_signal_header('ch1', sample_frequency=256)]
        sheaders[0]['physical_min'] = -200
        sheaders[0]['physical_max'] = 200
        with self.assertRaises(AssertionError):
            highlevel.write_edf(self.edfplus_data_file, signals, sheaders, digital=False)


    def test_read_write_accented(self):
        signals = np.random.rand(3, 256*60) # then rescale to 0-1
        signals = (signals - signals.min()) / (signals.max() - signals.min())
        highlevel.write_edf_quick(self.test_accented, signals, sfreq=256)
        signals2, _, _ = highlevel.read_edf(self.test_accented)

        np.testing.assert_allclose(signals, signals2, atol=0.00002)
        # if os.name!='nt':
        self.assertTrue(os.path.isfile(self.test_accented), 'File does not exist')

    def test_read_unicode(self):
        signals = np.random.rand(3, 256*60) # then rescale to 0-1
        signals = (signals - signals.min()) / (signals.max() - signals.min())
        success = highlevel.write_edf_quick(self.edfplus_data_file, signals, sfreq=256)
        self.assertTrue(success)
        shutil.copy(self.edfplus_data_file, self.test_unicode)
        signals2, _, _ = highlevel.read_edf(self.test_unicode)
        self.assertTrue(os.path.isfile(self.test_unicode), 'File does not exist')


    def test_read_header(self):

        header = highlevel.read_edf_header(self.test_generator)
        self.assertEqual(len(header), 15)
        self.assertEqual(len(header['channels']), 11)
        self.assertEqual(len(header['SignalHeaders']), 11)
        self.assertEqual(header['Duration'], 600)
        self.assertEqual(header['admincode'], 'Dr. X')
        self.assertEqual(header['birthdate'], '30 jun 1969')
        self.assertEqual(header['equipment'], 'test generator')
        self.assertEqual(header['sex'], 'Male')
        self.assertEqual(header['gender'], 'Male')
        self.assertEqual(header['patient_additional'], 'patient')
        self.assertEqual(header['patientcode'], 'abcxyz99')
        self.assertEqual(header['patientname'], 'Hans Muller')
        self.assertEqual(header['technician'], 'Mr. Spotty')


    def test_anonymize(self):

        header = highlevel.make_header(technician='tech', recording_additional='radd',
                                                patientname='name', patient_additional='padd',
                                                patientcode='42', equipment='eeg', admincode='420',
                                                sex='Male', birthdate='05.09.1980')
        annotations = [[0.01, -1, 'begin'],[0.5, -1, 'middle'],[10, -1, 'end']]
        header['annotations'] = annotations
        signal_headers = highlevel.make_signal_headers(['ch'+str(i) for i in range(3)])
        signals = np.random.rand(3, 256*300)*200 #5 minutes of eeg
        signals = (signals - signals.min()) / (signals.max() - signals.min())
        highlevel.write_edf(self.personalized, signals, signal_headers, header)



        highlevel.anonymize_edf(self.personalized, new_file=self.anonymized,
                                        to_remove=['patientname', 'birthdate',
                                                   'admincode', 'patientcode',
                                                   'technician'],
                                        new_values=['x', '', 'xx', 'xxx',
                                                    'xxxx'], verify=True, verbose=True)
        new_header = highlevel.read_edf_header(self.anonymized)
        self.assertEqual(new_header['birthdate'], '')
        self.assertEqual(new_header['patientname'], 'x')
        self.assertEqual(new_header['admincode'], 'xx')
        self.assertEqual(new_header['patientcode'], 'xxx')
        self.assertEqual(new_header['technician'], 'xxxx')


        highlevel.anonymize_edf(self.personalized, to_remove=['patientname', 'birthdate',
                                                   'admincode', 'patientcode',
                                                   'technician'],
                                        new_values=['x', '', 'xx', 'xxx',
                                                    'xxxx'], verify=True)
        new_header = highlevel.read_edf_header(self.personalized[:-4]+'_anonymized.edf')
        self.assertEqual(new_header['birthdate'], '')
        self.assertEqual(new_header['patientname'], 'x')
        self.assertEqual(new_header['admincode'], 'xx')
        self.assertEqual(new_header['patientcode'], 'xxx')
        self.assertEqual(new_header['technician'], 'xxxx')

        with self.assertRaises(AssertionError):
            highlevel.anonymize_edf(self.personalized,
                                    new_file=self.anonymized,
                                    to_remove=['patientname', 'birthdate',
                                               'admincode', 'patientcode',
                                               'technician'],
                                    new_values=['x', '', 'xx', 'xxx'],
                                    verify=True)

    def test_crop_edf(self):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        edf_file = os.path.join(data_dir, 'test_generator.edf')
        outfile = os.path.join(data_dir, 'tmp_test_generator_cropped.edf')
        orig_header = highlevel.read_edf_header(edf_file)  # noqa: E501
        orig_start = orig_header["startdate"]
        new_start = datetime(2011, 4, 4, 12, 58, 0)
        new_stop = datetime(2011, 4, 4, 13, 0, 0)

        # Test 1: no cropping
        # The output file should be the same as input.
        highlevel.crop_edf(
            edf_file, new_file=outfile, start=None, stop=None)
        assert highlevel.compare_edf(edf_file, outfile)

        # Test 2: crop using datetimes (default)
        # .. both start and stop
        highlevel.crop_edf(
            edf_file, new_file=outfile, start=new_start,
            stop=new_stop
        )
        # Test that the signal values are correctly cropped
        _compare_cropped_edf(edf_file, outfile)
        # .. only start
        highlevel.crop_edf(edf_file, new_file=outfile, start=new_start)
        _compare_cropped_edf(edf_file, outfile)
        # .. only stop
        highlevel.crop_edf(edf_file, new_file=outfile, stop=new_stop)
        _compare_cropped_edf(edf_file, outfile)

        # Test 3: crop using seconds
        new_start_sec = (new_start - orig_start).seconds
        new_stop_sec = (new_stop - orig_start).seconds
        # .. both start and stop
        highlevel.crop_edf(
            edf_file, new_file=outfile, start=new_start_sec,
            stop=new_stop_sec, start_format="seconds", stop_format="seconds"
        )
        _compare_cropped_edf(edf_file, outfile)
        # .. only start
        highlevel.crop_edf(
            edf_file, new_file=outfile,
            start=new_start_sec, start_format="seconds"
        )
        _compare_cropped_edf(edf_file, outfile)
        # .. only stop
        highlevel.crop_edf(
            edf_file, new_file=outfile, stop=new_stop_sec,
            stop_format="seconds"
        )
        _compare_cropped_edf(edf_file, outfile)

    def test_drop_channel(self):
        signal_headers = highlevel.make_signal_headers(['ch'+str(i) for i in range(5)])
        signals = np.random.rand(5, 256*300)*200 #5 minutes of eeg
        signals = (signals - signals.min()) / (signals.max() - signals.min())
        highlevel.write_edf(self.drop_from, signals, signal_headers)

        dropped = highlevel.drop_channels(self.drop_from, to_keep=['ch1', 'ch2'], verbose=True)

        signals2, signal_headers, header = highlevel.read_edf(dropped)

        np.testing.assert_allclose(signals[1:3,:], signals2, atol=0.01)

        highlevel.drop_channels(self.drop_from, self.drop_from[:-4]+'2.edf',
                                to_drop=['ch0', 'ch1', 'ch2'])
        signals2, signal_headers, header = highlevel.read_edf(self.drop_from[:-4]+'2.edf')

        np.testing.assert_allclose(signals[3:,:], signals2, atol=0.01)

        with self.assertRaises(AssertionError):
            highlevel.drop_channels(self.drop_from, to_keep=['ch1'], to_drop=['ch3'])


    def test_annotation_bytestring(self):
        header = highlevel.make_header(technician='tech', recording_additional='radd',
                                                patientname='name', patient_additional='padd',
                                                patientcode='42', equipment='eeg', admincode='420',
                                                sex='Male', birthdate='05.09.1980')
        annotations = [[0.01, b'-1', 'begin'],[0.5, b'-1', 'middle'],[10, -1, 'end']]
        header['annotations'] = annotations
        signal_headers = highlevel.make_signal_headers(['ch'+str(i) for i in range(3)])
        signals = np.random.rand(3, 256*300)*200 #5 minutes of eeg
        highlevel.write_edf(self.edfplus_data_file, signals, signal_headers, header)
        _,_,header2 = highlevel.read_edf(self.edfplus_data_file)
        highlevel.write_edf(self.edfplus_data_file, signals, signal_headers, header)
        _,_,header3 = highlevel.read_edf(self.edfplus_data_file)
        self.assertEqual(header2['annotations'], header3['annotations'])


if __name__ == '__main__':
    # run_module_suite(argv=sys.argv)
    unittest.main()
