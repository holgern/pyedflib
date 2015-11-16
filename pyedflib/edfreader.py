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
from datetime import datetime, date
import numpy as np
from ._pyedflib import *
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
        """
        annot = self.read_annotation()
        annot = np.array(annot)
        ann_time = self._get_float(annot[:, 0])
        ann_text = annot[:, 2]
        for i in np.arange(len(annot[:, 1])):
            if (annot[i, 1] == ''):
                annot[i, 1] = '-1'
        ann_duration = self._get_float(annot[:, 1])
        return ann_time/10000000, ann_duration, ann_text

    def _get_float(self,v):
        result = np.zeros(np.size(v))
        for i in np.arange(np.size(v)):
            try:
                if (not v[i]):
                    result[i] = -1
                else:
                    result[i] = int(v[i])
            except ValueError:
                result[i] = float(v[i])
        return result

    def getHeader(self):
        """
        Returns the file header as dict
        """
        return {"technician": self.getTechnician(), "recording_additional": self.getRecordingAdditional(),
                "patientname": self.getPatientName(), "patient_additional": self.getPatientAdditional(),
                "patientcode": self.getPatientCode(), "equipment": self.getEquipment(),
                "admincode": self.getAdmincode(), "gender": self.getGender(), "startdate": self.getStartdatetime(),
                "birthdate": self.getBirthdate()}

    def getSignalHeader(self,chn):
        """
        Returns the  header of one signal as  dicts
        """
        return {'label': self.getLabel(chn),
                                 'dimension': self.getPhysicalDimension(chn),
                                 'sample_rate': self.getSampleFrequency(chn),
                                 'physical_max':self.getPhysicialMaximum(chn),
                                 'physical_min': self.getPhysicialMinimum(chn),
                                 'digital_max': self.getDigitalMaximum(chn),
                                 'digital_min': self.getDigitalMinimum(chn),
                                 'prefilter':self.getPrefilter(chn),
                                 'transducer': self.getTransducer(chn)}

    def getSignalHeaders(self):
        """
        Returns the  header of all signals as array of dicts
        """
        signalHeader = []
        for chn in np.arange(self.n_channels):
            signalHeader.append(self.getSignalHeader(chn))
        return signalHeader

    def getTechnician(self):
        """
        Returns the technicians name
        """
        return self.technician.rstrip()

    def getRecordingAdditional(self):
        """
        Returns the additional recordinginfo
        """
        return self.recording_additional.rstrip()

    def getPatientName(self):
        """
        Returns the patientname
        """
        return self.patientname.rstrip()

    def getPatientCode(self):
        """
        Returns the patientcode
        """
        return self.patientcode.rstrip()

    def getPatientAdditional(self):
        """
        Returns the additional patientinfo.
        """
        return self.patient_additional.rstrip()

    def getEquipment(self):
        """
        Returns the used Equipment.
        """
        return self.equipment.rstrip()

    def getAdmincode(self):
        """
        Returns the Admincode.
        """
        return self.admincode.rstrip()

    def getGender(self):
        """
        Returns the Gender of the patient.
        """
        return self.gender.rstrip()

    def getFileDuration(self):
        """
        Returns the duration of the file in seconds.
        """
        return self.file_duration

    def getStartdatetime(self):
        """
        Returns the date and starttime as datetime object
        """
        return datetime(self.startdate_year, self.startdate_month, self.startdate_day,
                                 self.starttime_hour, self.starttime_minute, self.starttime_second)

    def getBirthdate(self):
        """
        Returns the birthdate as string object
        """
        return self.birthdate.rstrip()

    def getSampleFrequencies(self):
        """
        Returns  samplefrequencies of all signals.
        """
        return np.array([round(self.samplefrequency(chn))
                         for chn in np.arange(self.signals_in_file)])

    def getSampleFrequency(self,chn):
        """
        Returns the samplefrequency of signal edfsignal.
        """
        if (chn >= 0 and chn < self.signals_in_file):
            return round(self.samplefrequency(chn))
        else:
            return 0

    def getSignalLabels(self):
        """
        Returns all labels (name) ("FP1", "SaO2", etc.).
        """
        return [self.signal_label(chn).strip()
                for chn in np.arange(self.signals_in_file)]

    def getLabel(self,chn):
        """
        Returns the label (name) of signal chn ("FP1", "SaO2", etc.).
        """
        if (chn >= 0 and chn < self.signals_in_file):
            return self.signal_label(chn).rstrip()
        else:
            return b''

    def getPrefilter(self,chn):
        """
        Returns the prefilter of signal chn ("HP:0.1Hz", "LP:75Hz N:50Hz", etc.)
        """
        if (chn >= 0 and chn < self.signals_in_file):
            return self.prefilter(chn).rstrip()
        else:
            return b''

    def getPhysicialMaximum(self,chn):
        """
        Returns the maximum physical value of signal edfsignal.
        """
        if (chn >= 0 and chn < self.signals_in_file):
            return self.physical_max(chn)
        else:
            return 0

    def getPhysicialMinimum(self,chn):
        """
        Returns the minimum physical value of signal edfsignal.
        """
        if (chn >= 0 and chn < self.signals_in_file):
            return self.physical_min(chn)
        else:
            return 0

    def getDigitalMaximum(self,chn):
        """
        Returns the maximum digital value of signal edfsignal.
        """
        if (chn >= 0 and chn < self.signals_in_file):
            return self.digital_max(chn)
        else:
            return 0

    def getDigitalMinimum(self,chn):
        """
        Returns the minimum digital value of signal edfsignal.
        """
        if (chn >= 0 and chn < self.signals_in_file):
            return self.digital_min(chn)
        else:
            return 0

    def getTransducer(self,chn):
        """
        Returns the transducer of signal chn ("AgAgCl cup electrodes", etc.).
        """
        if (chn >= 0 and chn < self.signals_in_file):
            return self.transducer(chn).rstrip()
        else:
            return b''

    def getPhysicalDimension(self,chn):
        """
        Returns the physical dimension of signal edfsignal ("uV", "BPM", "mA", "Degr.", etc.)
        """
        if (chn >= 0 and chn < self.signals_in_file):
            return self.physical_dimension(chn).rstrip()
        else:
            return b''

    def readSignal(self, chn):

        nsamples = self.getNSamples()
        if (chn < len(nsamples)):
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
            print("label:", self.getSignalTextLabels()[ii], "fs:",
                  self.getSignalFreqs()[ii], "nsamples",
                  self.getNSamples()[ii])
