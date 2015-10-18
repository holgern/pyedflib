# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# Copyright (c) 2015 Chris Lee-Messer <https://bitbucket.org/cleemesser/>
# Copyright (c) 2015 Holger Nahrstaedt
from __future__ import division, print_function, absolute_import

__all__ = ['EdfReader']

from ._edflib import *
import numpy as np


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
        ann_time = annot[:, 0]
        ann_duration = annot[:, 0]
        ann_text = annot[:, 2]
        ann_time = ann_time.astype(np.float)
        ann_duration = ann_duration.astype(np.float)
        return ann_time, ann_duration, ann_text

    def getSignalFrequencies(self):
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
