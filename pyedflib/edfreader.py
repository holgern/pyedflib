# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# Copyright (c) 2015 Chris Lee-Messer <https://bitbucket.org/cleemesser/>
# Copyright (c) 2015 Holger Nahrstaedt
from __future__ import division, print_function, absolute_import

__all__ = ['EdfReader','Edfinfo']

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
        return annot
		
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
        
		

class Edfinfo(object):

    '''class to just get info about an edf file and print it
    just use the cython type to do so as we work on EdfReader'''
    
    def __init__(self, file_name):
        self.cedf = CyEdfReader(file_name)
        self.file_name = file_name
        self.signal_labels = []
        self.signal_nsamples = []
        self.samplefreqs = []
        ## do a lot of silly copying?
        self.signals_in_file = self.cedf.signals_in_file
        self.datarecords_in_file = self.cedf.datarecords_in_file
        for ii in xrange(self.signals_in_file):
            self.signal_labels.append(self.cedf.signal_label(ii))
            self.signal_nsamples.append(self.cedf.samples_in_file(ii))
            self.samplefreqs.append(self.cedf.samplefrequency(ii))

    def file_info(self):
        print("file name:", self.file_name)
        print("signals in file:", self.signals_in_file)

    def file_info_long(self):
        self.file_info()
        for ii in xrange(self.signals_in_file):
            print("label:", self.signal_labels[ii], "fs:", self.samplefreqs[ii], "nsamples", self.signal_nsamples[ii])

