# -*- coding: utf-8 -*-
# Copyright (c) 2015, Holger Nahrstaedt
# Copyright (c) 2011, 2015, Chris Lee-Messer
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
from __future__ import division, print_function, absolute_import
from datetime import datetime
import numpy as np

from ._extensions._pyedflib import CyEdfReader
__all__ = ['EdfReader']


class EdfReader(CyEdfReader):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, ex_tb):
        self._close()  # cleanup the file

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
        ann_time = self._get_float(annot[:, 0])
        ann_text = annot[:, 2]
        for i in np.arange(len(annot[:, 1])):
            ann_text[i] = self._convert_string(ann_text[i])
            if annot[i, 1] == '':
                annot[i, 1] = '-1'
        ann_duration = self._get_float(annot[:, 1])
        return ann_time/10000000, ann_duration, ann_text

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
        UNICODE_EXISTS = False
        try:
            UNICODE_EXISTS = bool(type(unicode))
        except NameError:
            # unicode = lambda s: str(s)
            UNICODE_EXISTS = False
        if UNICODE_EXISTS:
            return unicode(s)
        else:
            return s.decode("utf-8", "strict")

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
                "admincode": self.getAdmincode(), "gender": self.getGender(), "startdate": self.getStartdatetime(),
                "birthdate": self.getBirthdate()}

    def getSignalHeader(self, chn):
        """
        Returns the  header of one signal as  dicts

        Parameters
        ----------
        None
        """
        return {'label': self.getLabel(chn),
                'dimension': self.getPhysicalDimension(chn),
                                 'sample_rate': self.getSampleFrequency(chn),
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
        for chn in np.arange(self.n_channels):
            signalHeader.append(self.getSignalHeader(chn))
        return signalHeader

    def getTechnician(self):
        """
        Returns the technicians name

        Parameters
        ----------
        None

        Examples
        --------
        >>> import pyedflib
        >>> f = pyedflib.data.test_generator()
        >>> f.getTechnician()==''
        True
        >>> f._close()
        >>> del f

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
        >>> f._close()
        >>> del f

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
        >>> f._close()
        >>> del f

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
        >>> f._close()
        >>> del f

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
        >>> f._close()
        >>> del f

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
        >>> f._close()
        >>> del f

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
        >>> f._close()
        >>> del f

        """
        return self._convert_string(self.admincode.rstrip())

    def getGender(self):
        """
        Returns the Gender of the patient.

        Parameters
        ----------
        None

        Examples
        --------
        >>> import pyedflib
        >>> f = pyedflib.data.test_generator()
        >>> f.getGender()==''
        True
        >>> f._close()
        >>> del f

        """
        return self._convert_string(self.gender.rstrip())

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
        >>> f._close()
        >>> del f

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
        >>> f._close()
        >>> del f

        """
        return datetime(self.startdate_year, self.startdate_month, self.startdate_day,
                                 self.starttime_hour, self.starttime_minute, self.starttime_second)

    def getBirthdate(self):
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
        >>> f._close()
        >>> del f

        """
        return self._convert_string(self.birthdate.rstrip())

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
        >>> f._close()
        >>> del f

        """
        return np.array([round(self.samplefrequency(chn))
                         for chn in np.arange(self.signals_in_file)])

    def getSampleFrequency(self,chn):
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
        >>> f._close()
        >>> del f

        """
        if 0 <= chn < self.signals_in_file:
            return round(self.samplefrequency(chn))
        else:
            return 0

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
        >>> f._close()
        >>> del f

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
        >>> f._close()
        >>> del f

        """
        if 0 <= chn < self.signals_in_file:
            return self._convert_string(self.signal_label(chn).rstrip())
        else:
            return self._convert_string(b'')

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
        >>> f._close()
        >>> del f

        """
        if 0 <= chn < self.signals_in_file:
            return self._convert_string(self.prefilter(chn).rstrip())
        else:
            return self._convert_string(b'')

    def getPhysicalMaximum(self,chn):
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
        >>> f._close()
        >>> del f

        """
        if 0 <= chn < self.signals_in_file:
            return self.physical_max(chn)
        else:
            return 0

    def getPhysicalMinimum(self,chn):
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
        >>> f._close()
        >>> del f

        """
        if 0 <= chn < self.signals_in_file:
            return self.physical_min(chn)
        else:
            return 0

    def getDigitalMaximum(self, chn):
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
        >>> f._close()
        >>> del f

        """
        if 0 <= chn < self.signals_in_file:
            return self.digital_max(chn)
        else:
            return 0

    def getDigitalMinimum(self, chn):
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
        >>> f._close()
        >>> del f

        """
        if 0 <= chn < self.signals_in_file:
            return self.digital_min(chn)
        else:
            return 0

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
        >>> f._close()
        >>> del f

        """
        if 0 <= chn < self.signals_in_file:
            return self._convert_string(self.transducer(chn).rstrip())
        else:
            return self._convert_string('')

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
        >>> f._close()
        >>> del f

        """
        if 0 <= chn < self.signals_in_file:
            return self._convert_string(self.physical_dimension(chn).rstrip())
        else:
            return self._convert_string(b'')

    def readSignal(self, chn):

        nsamples = self.getNSamples()
        if chn < len(nsamples):
            x = np.zeros(nsamples[chn], dtype=np.float64)

            v = x[chn*nsamples[chn]:(chn+1)*nsamples[chn]]
            self.readsignal(chn, 0, nsamples[chn], v)
            return x
        else:
            return np.array([])

    def file_info(self):
        print("file name:", self.file_name)
        print("signals in file:", self.signals_in_file)

    def file_info_long(self):
        self.file_info()
        for ii in np.arange(self.signals_in_file):
            print("label:", self.getSignalLabel(ii), "fs:",
                  self.getSignalFreqs()[ii], "nsamples",
                  self.getNSamples()[ii])
