# Copyright (c) 2019 - 2023 Simon Kern
# Copyright (c) 2015 - 2023 Holger Nahrstaedt
# Copyright (c) 2011, 2015, Chris Lee-Messer
# Copyright (c) 2016-2017 The pyedflib Developers
#                         <https://github.com/holgern/pyedflib>
# See LICENSE for license details.

from datetime import datetime
import numpy as np
import warnings

from ._extensions._pyedflib import CyEdfReader

__all__ = ['EdfReader', 'DO_NOT_READ_ANNOTATIONS',
           'READ_ANNOTATIONS', 'READ_ALL_ANNOTATIONS', 'CHECK_FILE_SIZE',
           'DO_NOT_CHECK_FILE_SIZE', 'REPAIR_FILE_SIZE_IF_WRONG']

DO_NOT_READ_ANNOTATIONS = 0
READ_ANNOTATIONS = 1
READ_ALL_ANNOTATIONS = 2

CHECK_FILE_SIZE = 0
DO_NOT_CHECK_FILE_SIZE = 1
REPAIR_FILE_SIZE_IF_WRONG = 2

def _debug_parse_header(filename): #pragma: no cover
    """
    A debug function that reads a header and outputs everything that
    is contained in the header
    """
    import json
    from collections import OrderedDict
    header = OrderedDict()
    with open(filename, 'rb') as f:
        f.seek(0)
        header['version'] = f.read(8).decode()
        header['patient_id'] = f.read(80).decode().strip()
        header['recording_id'] = f.read(80).decode().strip()
        header['startdate'] = f.read(8).decode()
        header['starttime'] = f.read(8).decode()
        header['header_n_bytes'] = f.read(8).decode()
        header['reserved'] = f.read(44).decode().strip()
        header['n_records'] = f.read(8).decode()
        header['record_duration'] = f.read(8).decode()
        header['n_signals'] = f.read(4).decode()

        print('\n##### Header')
        print(json.dumps(header, indent=2))

        nsigs = int(header['n_signals'])
        label = [f.read(16).decode() for i in range(nsigs)]
        transducer = [f.read(80).decode().strip() for i in range(nsigs)]
        dimension = [f.read(8).decode().strip() for i in range(nsigs)]
        pmin = [f.read(8).decode() for i in range(nsigs)]
        pmax = [f.read(8).decode() for i in range(nsigs)]
        dmin = [f.read(8).decode() for i in range(nsigs)]
        dmax = [f.read(8).decode() for i in range(nsigs)]
        prefilter = [f.read(80).decode().strip() for i in range(nsigs)]
        n_samples = [f.read(8).decode() for i in range(nsigs)]
        reserved = [f.read(32).decode() for i in range(nsigs)]
    _ = zip(label, transducer, dimension, pmin, pmax, dmin, dmax, prefilter, n_samples, reserved)
    values = locals().copy()
    fields = ['label', 'transducer', 'dimension', 'pmin', 'pmax', 'dmin', 'dmax', 'prefilter', 'n_samples', 'reserved']
    sheaders = [{field:values[field][i] for field in fields} for i in range(nsigs)]
    print('\n##### Signal Headers')
    print(json.dumps(sheaders, indent=2))


class EdfReader(CyEdfReader):
    """
    This provides a simple interface to read EDF, EDF+, BDF and BDF+ files.
    """
    def __enter__(self):
        return self

    def __del__(self):
        self._close()

    def __exit__(self, exc_type, exc_val, ex_tb):
        self._close()  # cleanup the file

    def close(self):
        """
        Closes the file handler
        """
        self._close()

    def getNSamples(self):
        return np.array([self.samples_in_file(chn)
                         for chn in np.arange(self.signals_in_file)])

    def readAnnotations(self):
        """
        Annotations from a edf-file

        Parameters
        ----------
        None
        """
        annot = self.read_annotation()
        annot = np.array(annot)
        if (annot.shape[0] == 0):
            return np.array([]), np.array([]), np.array([])
        ann_time = self._get_float(annot[:, 0])
        ann_text = annot[:, 2]
        ann_text_out = ["" for x in range(len(annot[:, 1]))]
        for i in np.arange(len(annot[:, 1])):
            ann_text_out[i] = self._convert_string(ann_text[i])
            if annot[i, 1] == '':
                annot[i, 1] = '-1'
        ann_duration = self._get_float(annot[:, 1])
        return ann_time/10000000, ann_duration, np.array(ann_text_out)

    def _get_float(self, v):
        result = np.zeros(np.size(v))
        for i in np.arange(np.size(v)):
            try:
                if not v[i]:
                    result[i] = -1
                else:
                    result[i] = int(v[i])
            except ValueError:
                result[i] = float(v[i])
        return result

    def _convert_string(self,s):
        if isinstance(s, bytes):
            return s.decode("latin")
        else:
            return s.decode("utf_8", "strict")

    def getHeader(self):
        """
        Returns the file header as dict

        Parameters
        ----------
        None
        """
        return {"technician": self.getTechnician(), "recording_additional": self.getRecordingAdditional(),
                "patientname": self.getPatientName(), "patient_additional": self.getPatientAdditional(),
                "patientcode": self.getPatientCode(), "equipment": self.getEquipment(),
                "admincode": self.getAdmincode(), "sex": self.getSex(), "startdate": self.getStartdatetime(),
                "birthdate": self.getBirthdate(),
                "gender": self.getSex()}  # backwards compatibility

    def getSignalHeader(self, chn):
        """
        Returns the  header of one signal as  dicts

        Parameters
        ----------
        None
        """
        return {'label': self.getLabel(chn),
                'dimension': self.getPhysicalDimension(chn),
                'sample_rate': self.getSampleFrequency(chn),  # backwards compatibility
                'sample_frequency': self.getSampleFrequency(chn),
                'physical_max':self.getPhysicalMaximum(chn),
                'physical_min': self.getPhysicalMinimum(chn),
                'digital_max': self.getDigitalMaximum(chn),
                'digital_min': self.getDigitalMinimum(chn),
                'prefilter':self.getPrefilter(chn),
                'transducer': self.getTransducer(chn)}

    def getSignalHeaders(self):
        """
        Returns the  header of all signals as array of dicts

        Parameters
        ----------
        None
        """
        signalHeader = []
        for chn in np.arange(self.signals_in_file):
            signalHeader.append(self.getSignalHeader(chn))
        return signalHeader

    def getTechnician(self):
        """
        Returns the technicians name

        Parameters
        ----------
        None

        .. code-block:: python

            >>> import pyedflib
            >>> f = pyedflib.data.test_generator()
            >>> f.getTechnician()==''
            True
            >>> f.close()

        """
        return self._convert_string(self.technician.rstrip())

    def getRecordingAdditional(self):
        """
        Returns the additional recordinginfo

        Parameters
        ----------
        None

        Examples
        --------
        >>> import pyedflib
        >>> f = pyedflib.data.test_generator()
        >>> f.getRecordingAdditional()==''
        True
        >>> f.close()

        """
        return self._convert_string(self.recording_additional.rstrip())

    def getPatientName(self):
        """
        Returns the patientname

        Parameters
        ----------
        None

        Examples
        --------
        >>> import pyedflib
        >>> f = pyedflib.data.test_generator()
        >>> f.getPatientName()=='X'
        True
        >>> f.close()

        """
        return self._convert_string(self.patientname.rstrip())

    def getPatientCode(self):
        """
        Returns the patientcode

        Parameters
        ----------
        None

        Examples
        --------
        >>> import pyedflib
        >>> f = pyedflib.data.test_generator()
        >>> f.getPatientCode()==''
        True
        >>> f.close()

        """
        return self._convert_string(self.patientcode.rstrip())

    def getPatientAdditional(self):
        """
        Returns the additional patientinfo.

        Parameters
        ----------
        None

        Examples
        --------
        >>> import pyedflib
        >>> f = pyedflib.data.test_generator()
        >>> f.getPatientAdditional()==''
        True
        >>> f.close()

        """
        return self._convert_string(self.patient_additional.rstrip())

    def getEquipment(self):
        """
        Returns the used Equipment.

        Parameters
        ----------
        None

        Examples
        --------
        >>> import pyedflib
        >>> f = pyedflib.data.test_generator()
        >>> f.getEquipment()=='test generator'
        True
        >>> f.close()

        """
        return self._convert_string(self.equipment.rstrip())

    def getAdmincode(self):
        """
        Returns the Admincode.

        Parameters
        ----------
        None

        Examples
        --------
        >>> import pyedflib
        >>> f = pyedflib.data.test_generator()
        >>> f.getAdmincode()==''
        True
        >>> f.close()

        """
        return self._convert_string(self.admincode.rstrip())

    def getSex(self):
        """
        Returns the Sex of the patient.

        Parameters
        ----------
        None

        Examples
        --------
        >>> import pyedflib
        >>> f = pyedflib.data.test_generator()
        >>> f.getSex()==''
        True
        >>> f.close()

        """
        return self._convert_string(self.sex.rstrip())

    def getGender(self):
        warnings.warn("Method 'getGender' is deprecated, use 'getSex' instead.", DeprecationWarning, stacklevel=2)
        return self.getSex()

    def getFileDuration(self):
        """
        Returns the duration of the file in seconds.

        Parameters
        ----------
        None

        Examples
        --------
        >>> import pyedflib
        >>> f = pyedflib.data.test_generator()
        >>> f.getFileDuration()==600
        True
        >>> f.close()

        """
        return self.file_duration

    def getStartdatetime(self):
        """
        Returns the date and starttime as datetime object

        Parameters
        ----------
        None

        Examples
        --------
        >>> import pyedflib
        >>> f = pyedflib.data.test_generator()
        >>> f.getStartdatetime()
        datetime.datetime(2011, 4, 4, 12, 57, 2)
        >>> f.close()

        """
        # denoted as long long in nanoseconds, we need to transfer it to microsecond
        subsecond = np.round(self.starttime_subsecond/100).astype(int)
        return datetime(self.startdate_year, self.startdate_month, self.startdate_day,
                                 self.starttime_hour, self.starttime_minute, self.starttime_second,
                                 subsecond)

    def getBirthdate(self, string=True):
        """
        Returns the birthdate as string object

        Parameters
        ----------
        None

        Examples
        --------
        >>> import pyedflib
        >>> f = pyedflib.data.test_generator()
        >>> f.getBirthdate()=='30 jun 1969'
        True
        >>> f.close()

        """

        if string:
            return self._convert_string(self.birthdate.rstrip())
        else:
            return datetime.strptime(self._convert_string(self.birthdate.rstrip()), "%d %b %Y")

    def getSampleFrequencies(self):
        """
        Returns  samplefrequencies of all signals.

        Parameters
        ----------
        None

        Examples
        --------
        >>> import pyedflib
        >>> f = pyedflib.data.test_generator()
        >>> all(f.getSampleFrequencies()==200.0)
        True
        >>> f.close()

        """
        return np.array([self.samplefrequency(chn)
                         for chn in np.arange(self.signals_in_file)])

    def getSampleFrequency(self, chn):
        """
        Returns the samplefrequency of signal edfsignal.

        Parameters
        ----------
        chn : int
            channel number

        Examples
        --------
        >>> import pyedflib
        >>> f = pyedflib.data.test_generator()
        >>> f.getSampleFrequency(0)==200.0
        True
        >>> f.close()

        """
        if 0 <= chn < self.signals_in_file:
            return self.samplefrequency(chn)
        else:
            raise IndexError('Trying to access channel {}, but only {} ' \
                             'channels found'.format(chn, self.signals_in_file))

    def getSignalLabels(self):
        """
        Returns all labels (name) ("FP1", "SaO2", etc.).

        Parameters
        ----------
        None

        Examples
        --------
        >>> import pyedflib
        >>> f = pyedflib.data.test_generator()
        >>> f.getSignalLabels()==['squarewave', 'ramp', 'pulse', 'noise', 'sine 1 Hz', 'sine 8 Hz', 'sine 8.1777 Hz', 'sine 8.5 Hz', 'sine 15 Hz', 'sine 17 Hz', 'sine 50 Hz']
        True
        >>> f.close()

        """
        return [self._convert_string(self.signal_label(chn).strip())
                for chn in np.arange(self.signals_in_file)]

    def getLabel(self,chn):
        """
        Returns the label (name) of signal chn ("FP1", "SaO2", etc.).

        Parameters
        ----------
        chn : int
            channel number

        Examples
        --------
        >>> import pyedflib
        >>> f = pyedflib.data.test_generator()
        >>> f.getLabel(0)=='squarewave'
        True
        >>> f.close()

        """
        if 0 <= chn < self.signals_in_file:
            return self._convert_string(self.signal_label(chn).rstrip())
        else:
            raise IndexError('Trying to access channel {}, but only {} ' \
                             'channels found'.format(chn, self.signals_in_file))

    def getPrefilter(self,chn):
        """
        Returns the prefilter of signal chn ("HP:0.1Hz", "LP:75Hz N:50Hz", etc.)

        Parameters
        ----------
        chn : int
            channel number

        Examples
        --------
        >>> import pyedflib
        >>> f = pyedflib.data.test_generator()
        >>> f.getPrefilter(0)==''
        True
        >>> f.close()

        """
        if 0 <= chn < self.signals_in_file:
            return self._convert_string(self.prefilter(chn).rstrip())
        else:
            raise IndexError('Trying to access channel {}, but only {} ' \
                             'channels found'.format(chn, self.signals_in_file))

    def getPhysicalMaximum(self,chn=None):
        """
        Returns the maximum physical value of signal edfsignal.

        Parameters
        ----------
        chn : int
            channel number

        Examples
        --------
        >>> import pyedflib
        >>> f = pyedflib.data.test_generator()
        >>> f.getPhysicalMaximum(0)==1000.0
        True
        >>> f.close()

        """
        if chn is not None:
            if 0 <= chn < self.signals_in_file:
                return self.physical_max(chn)
            else:
                raise IndexError('Trying to access channel {}, but only {} ' \
                             'channels found'.format(chn, self.signals_in_file))
        else:
            physMax = np.zeros(self.signals_in_file)
            for i in np.arange(self.signals_in_file):
                physMax[i] = self.physical_max(i)
            return physMax

    def getPhysicalMinimum(self,chn=None):
        """
        Returns the minimum physical value of signal edfsignal.

        Parameters
        ----------
        chn : int
            channel number

        Examples
        --------
        >>> import pyedflib
        >>> f = pyedflib.data.test_generator()
        >>> f.getPhysicalMinimum(0)==-1000.0
        True
        >>> f.close()

        """
        if chn is not None:
            if 0 <= chn < self.signals_in_file:
                return self.physical_min(chn)
            else:
                raise IndexError('Trying to access channel {}, but only {} ' \
                             'channels found'.format(chn, self.signals_in_file))
        else:
            physMin = np.zeros(self.signals_in_file)
            for i in np.arange(self.signals_in_file):
                physMin[i] = self.physical_min(i)
            return physMin

    def getDigitalMaximum(self, chn=None):
        """
        Returns the maximum digital value of signal edfsignal.

        Parameters
        ----------
        chn : int
            channel number

        Examples
        --------
        >>> import pyedflib
        >>> f = pyedflib.data.test_generator()
        >>> f.getDigitalMaximum(0)
        32767
        >>> f.close()

        """
        if chn is not None:
            if 0 <= chn < self.signals_in_file:
                return self.digital_max(chn)
            else:
                raise IndexError('Trying to access channel {}, but only {} ' \
                             'channels found'.format(chn, self.signals_in_file))
        else:
            digMax = np.zeros(self.signals_in_file)
            for i in np.arange(self.signals_in_file):
                digMax[i] = self.digital_max(i)
            return digMax

    def getDigitalMinimum(self, chn=None):
        """
        Returns the minimum digital value of signal edfsignal.

        Parameters
        ----------
        chn : int
            channel number

        Examples
        --------
        >>> import pyedflib
        >>> f = pyedflib.data.test_generator()
        >>> f.getDigitalMinimum(0)
        -32768
        >>> f.close()

        """
        if chn is not None:
            if 0 <= chn < self.signals_in_file:
                return self.digital_min(chn)
            else:
                raise IndexError('Trying to access channel {}, but only {} ' \
                             'channels found'.format(chn, self.signals_in_file))
        else:
            digMin = np.zeros(self.signals_in_file)
            for i in np.arange(self.signals_in_file):
                digMin[i] = self.digital_min(i)
            return digMin

    def getTransducer(self, chn):
        """
        Returns the transducer of signal chn ("AgAgCl cup electrodes", etc.).

        Parameters
        ----------
        chn : int
            channel number

        Examples
        --------
        >>> import pyedflib
        >>> f = pyedflib.data.test_generator()
        >>> f.getTransducer(0)==''
        True
        >>> f.close()

        """
        if 0 <= chn < self.signals_in_file:
            return self._convert_string(self.transducer(chn).rstrip())
        else:
            raise IndexError('Trying to access channel {}, but only {} ' \
                             'channels found'.format(chn, self.signals_in_file))
    def getPhysicalDimension(self, chn):
        """
        Returns the physical dimension of signal edfsignal ("uV", "BPM", "mA", "Degr.", etc.)

        Parameters
        ----------
        chn : int
            channel number

        Examples
        --------
        >>> import pyedflib
        >>> f = pyedflib.data.test_generator()
        >>> f.getPhysicalDimension(0)=='uV'
        True
        >>> f.close()

        """
        if 0 <= chn < self.signals_in_file:
            return self._convert_string(self.physical_dimension(chn).rstrip())
        else:
            raise IndexError('Trying to access channel {}, but only {} ' \
                             'channels found'.format(chn, self.signals_in_file))

    def readSignal(self, chn, start=0, n=None, digital=False):
        """
        Returns the physical data of signal chn. When start and n is set, a subset is returned

        Parameters
        ----------
        chn : int
            channel number
        start : int
            start pointer (default is 0)
        n : int
            length of data to read (default is None, by which the complete data of the channel are returned)
        digital: bool
            will return the signal in original digital values instead of physical values
        Examples
        --------
        >>> import pyedflib
        >>> f = pyedflib.data.test_generator()
        >>> x = f.readSignal(0,0,1000)
        >>> int(x.shape[0])
        1000
        >>> x2 = f.readSignal(0)
        >>> int(x2.shape[0])
        120000
        >>> f.close()

        """
        if start < 0:
            return np.array([])
        if n is not None and n < 0:
            return np.array([])
        nsamples = self.getNSamples()
        if 0 <= chn < len(nsamples):
            if n is None:
                n = nsamples[chn]
            elif n > nsamples[chn]:
                return np.array([])
            dtype = np.int32 if digital else np.float64
            x = np.zeros(n, dtype=dtype)
            if digital:
                self.read_digital_signal(chn, start, n, x)
            else:
                self.readsignal(chn, start, n, x)
            return x
        else:
            raise IndexError('Trying to access channel {}, but only {} ' \
                             'channels found'.format(chn, self.signals_in_file))

    def file_info(self):
        print("file name:", self.file_name)
        print("signals in file:", self.signals_in_file)

    def file_info_long(self):
        """
        Returns information about the opened EDF/BDF file
        """
        self.file_info()
        for ii in np.arange(self.signals_in_file):
            print("label:", self.getSignalLabels()[ii], "fs:",
                  self.getSampleFrequencies()[ii], "nsamples",
                  self.getNSamples()[ii])
