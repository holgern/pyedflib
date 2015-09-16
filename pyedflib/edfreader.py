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
        self._close() # cleanup the file 
        
    def getNSamples(self):
        return np.array([self.samples_in_file(chn) for chn in xrange(self.signals_in_file)])
        
    def readAnnotations(self):
        annot = self.read_annotation()
        annot = np.array(annot)
        ann_time = annot[:,0]
        ann_duration = annot[:,0]
        ann_text = annot[:,2]
        ann_time = ann_time.astype(np.float)        
        ann_duration = ann_duration.astype(np.float)
        return ann_time, ann_duration, ann_text
		
    def getSignalFreqs(self):
        return np.array([self.samplefrequency(chn) for chn in xrange(self.signals_in_file)])
		
    def getSignalTextLabels(self):
        return  [self.signal_label(chn).strip() for chn in xrange(self.signals_in_file)]
    
    def readSignal(self,chn):

        nsamples = self.getNSamples()
        if (chn < len(nsamples)):
            x = np.zeros(nsamples[chn], dtype=np.float64)
            
            v = x[chn*nsamples[chn]:(chn+1)*nsamples[chn]]
            self.readsignal(chn, 0, nsamples[chn],v)
            return x
        else:
            return np.array([])
    def file_info(self):
        print("file name:", self.file_name)
        print("signals in file:", self.signals_in_file) 
        
    def file_info_long(self):
        self.file_info()
        for ii in xrange(self.signals_in_file):
            print("label:", self.getSignalTextLabels()[ii], "fs:", self.getSignalFreqs()[ii], "nsamples", self.getNSamples()[ii])
    
		
