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

__doc__ = """Pyrex wrapper for low-level C edflib implementation."""
__all__ = ['lib_version', 'CyEdfReader', 'set_patientcode', 
           'write_annotation_latin1', 'write_annotation_utf8', 'set_technician', 'EdfAnnotation',
           'get_annotation', 'read_int_samples', 'blockwrite_digital_samples', 'blockwrite_physical_samples',
           'set_recording_additional', 'write_physical_samples' ,'set_patientname', 'set_physical_minimum', 
           'read_physical_samples', 'close_file', 'set_physical_maximum', 'open_file_writeonly', 
           'set_patient_additional', 'set_digital_maximum', 'set_birthdate', 'set_digital_minimum',
           'write_digital_samples', 'set_equipment', 'set_samplefrequency','set_admincode', 'set_label',
           'tell', 'rewind', 'set_gender','set_physical_dimension', 'set_transducer', 'set_prefilter',
           'seek', 'set_startdatetime' ,'set_datarecord_duration', 'open_errors', 'FILETYPE_EDFPLUS',
           'FILETYPE_EDF','FILETYPE_BDF','FILETYPE_BDFPLUS']


#from c_edf cimport *
cimport c_edf
cimport cpython
import numpy as np
cimport numpy as np
from datetime import datetime, date
from cpython.version cimport PY_MAJOR_VERSION
include "edf.pxi"

open_errors = {
    EDFLIB_MALLOC_ERROR : "malloc error",                                                
    EDFLIB_NO_SUCH_FILE_OR_DIRECTORY   : "can not open file, no such file or directory",
    EDFLIB_FILE_CONTAINS_FORMAT_ERRORS : "the file is not EDF(+) or BDF(+) compliant (it contains format errors)",
    EDFLIB_MAXFILES_REACHED            : "to many files opened",
    EDFLIB_FILE_READ_ERROR             : "a read error occurred",
    EDFLIB_FILE_ALREADY_OPENED         : "file has already been opened",
    'default' : "unknown error"
    }

# constants are redeclared here so we can access them from Python
FILETYPE_EDF = EDFLIB_FILETYPE_EDF
FILETYPE_EDFPLUS = EDFLIB_FILETYPE_EDFPLUS
FILETYPE_BDF = EDFLIB_FILETYPE_BDF
FILETYPE_BDFPLUS = EDFLIB_FILETYPE_BDFPLUS


def lib_version():
    return c_edf.edflib_version()

cdef class CyEdfReader:
    """
    This provides a simple interface to read EDF, EDF+, and probably is ok with
    BDF and BDF+ files
    Note that edflib.c is encapsulated so there is no direct access to the file
    from here unless I add a raw interface or something

    EDF/BDF+ files are arranged into N signals sampled at rate Fs. The data is actually stored in chunks called    "datarecords" which have a file specific size.

    A typical way to use this to read an EEG file would be to choose a certain
    number of seconds per page to display. Then figureout how many data records
    that is. Then read in that many data records at a time. Transform the data as
    needed according the montage and filter settings, then display the data.

    """


    cdef int handle
    cdef c_edf.edf_hdr_struct hdr
    cdef size_t nsamples_per_record
    #I think it is ok not to do this in __cinit__(*,**)
    def __init__(self, file_name, annotations_mode='all'):
        self.open(file_name, mode='r', annotations_mode=annotations_mode)

    def __dealloc__(self):
        if self.hdr.handle:
            c_edf.edfclose_file(self.hdr.handle)
            
    def check_open_ok(self,result):
        if result == 0:
            return True
        else:
            raise IOError, open_errors[result]
            # return False
            
    def make_buffer(self):
        """
        utilty function to make a buffer that can hold a single datarecord. This will
        hold the physical samples for a single data record as a numpy tensor.

        -  might extend to provide for N datarecord size

        """
        print "self.hdr.datarecords_in_file", self.hdr.datarecords_in_file
        tmp =0
        for ii in range(self.signals_in_file):
            tmp += self.samples_in_datarecord(ii)
        self.nsamples_per_record = tmp 
        dbuffer = np.zeros(tmp, dtype='float64') # will get physical samples, not the orignal digital samples
        return dbuffer
    
    def open(self, file_name, mode='r', annotations_mode='all'):
        file_name_str = file_name.encode()
        result = c_edf.edfopen_file_readonly(file_name_str, &self.hdr, EDFLIB_READ_ALL_ANNOTATIONS)
        self.file_name = file_name
        return self.check_open_ok(result)

    def read_annotation(self):
        cdef c_edf.edf_annotation_struct annot
        annotlist = [['','',''] for x in range(self.annotations_in_file)]
        for ii in range(self.annotations_in_file):
            c_edf.edf_get_annotation(self.hdr.handle, ii, &(annot))
            #get_annotation(self.hdr.handle, ii, &annotation)
            annotlist[ii][0] = annot.onset
            annotlist[ii][1] = annot.duration
            annotlist[ii][2] = annot.annotation
        return annotlist



    property handle:
        "edflib internal int handle"
        def __get__(self):
            return self.handle
        
    property datarecords_in_file:
        "number of data records"
        def __get__(self):
            return self.hdr.datarecords_in_file
            


    property signals_in_file:
        def __get__(self):
            return self.hdr.edfsignals

    property file_duration:
        "file duration in seconds"
        def __get__(self):
            return self.hdr.file_duration/EDFLIB_TIME_DIMENSION

    property patient:
        "patient name?"
        def __get__(self):
            return self.hdr.patient

    property datarecord_duration:
        "datarecord duration in seconds (as a double)"
        def __get__(self):
            return (<double>self.hdr.datarecord_duration) / EDFLIB_TIME_DIMENSION

    property annotations_in_file:
        def __get__(self):
            return self.hdr.annotations_in_file

    property patientcode:
        def __get__(self):
            return self.hdr.patientcode

    property gender:
        def __get__(self):
            return self.hdr.gender

    property birthdate:
        def __get__(self):
            return self.hdr.birthdate

    property patientname:
        def __get__(self):
            return self.hdr.patient_name

    property patient_additional:
        def __get__(self):
            return self.hdr.patient_additional

    property startdate_year:
        def __get__(self):
            return self.hdr.startdate_year

    property startdate_month:
        def __get__(self):
            return self.hdr.startdate_month

    property startdate_day:
        def __get__(self):
            return self.hdr.startdate_day

    property starttime_hour:
        def __get__(self):
            return self.hdr.starttime_hour

    property starttime_minute:
        def __get__(self):
            return self.hdr.starttime_minute

    property starttime_second:
        def __get__(self):
            return self.hdr.starttime_second

    property admincode:
        def __get__(self):
            return self.hdr.admincode

    property technician:
        def __get__(self):
            return self.hdr.technician

    property equipment:
        def __get__(self):
            return self.hdr.equipment

    property recording_additional:
        def __get__(self):
            return self.hdr.recording_additional

    # signal parameters
    def signal_label(self, channel):
        return self.hdr.signalparam[channel].label

    def samples_in_file(self,channel):
        return self.hdr.signalparam[channel].smp_in_file

    def samples_in_datarecord(self, channel):
        return self.hdr.signalparam[channel].smp_in_datarecord

    def physical_dimension(self, channel):
        return self.hdr.signalparam[channel].physdimension

    def physical_max(self, channel):
        return self.hdr.signalparam[channel].phys_max

    def physical_min(self, channel):
        return self.hdr.signalparam[channel].phys_min

    def digital_max(self, channel):
        return self.hdr.signalparam[channel].dig_max    

    def digital_min(self, channel):
        return self.hdr.signalparam[channel].dig_min

    def prefilter(self, channel):
        return self.hdr.signalparam[channel].prefilter

    def transducer(self, channel):
        return self.hdr.signalparam[channel].transducer

    def samplefrequency(self, channel):
        return (<double>self.hdr.signalparam[channel].smp_in_datarecord / self.hdr.datarecord_duration) * EDFLIB_TIME_DIMENSION

    # def _tryoffset0(self):
    #     """
    #     fooling around to find offset in file to allow shortcut mmap interface
    #     """
    #     # cdef long offset = self.hdr.hdrsize  # from edflib.c read_physical_samples()
    #     print "trying to find data offset in file"
    #     nrecords = self.hdr.datarecords_in_file
    #     print "nrecords in file:", nrecords
    #     return 1,2
    #     # return offset, nrecords
    #     # print "offset via edftell:",  edftell(self.hdr.handle, 0)
        

    def _close(self):   # should not be closed from python
        c_edf.edfclose_file(self.hdr.handle)
    
    def read_digital_signal(self, signalnum, start, n, np.ndarray[np.int32_t, ndim=1] sigbuf):
       """read @n number of samples from signal number @signum starting at @start
          into numpy int32 array @sigbuf sigbuf must be at least n long
       """
       c_edf.edfseek(self.hdr.handle, signalnum, start, EDFSEEK_SET)
       readn = read_int_samples(self.handle, signalnum, n, sigbuf)
       if readn != n:
           print "read %d, less than %d requested!!!" % (readn, n)    

    def readsignal(self, signalnum, start, n, np.ndarray[np.float64_t, ndim=1] sigbuf):

        """read @n number of samples from signal number @signum starting at
        @start into numpy float64 array @sigbuf sigbuf must be at least n long
        """
        
        c_edf.edfseek(self.hdr.handle, signalnum, start, EDFSEEK_SET)
        readn = c_edf.edfread_physical_samples(self.hdr.handle, signalnum, n, <double*>sigbuf.data)
        # print "read %d samples" % readn
        if readn != n:
            print "read %d, less than %d requested!!!" % (readn, n)
        
    def load_datarecord(self, np.ndarray[np.float64_t, ndim=1] db, n=0):
        cdef size_t offset =0

        if n < self.hdr.datarecords_in_file:
            for ii in range(self.signals_in_file):
                c_edf.edfseek(self.hdr.handle, ii, n*self.samples_in_datarecord(ii), EDFSEEK_SET) # just a guess
                readn = c_edf.edfread_physical_samples(self.hdr.handle, ii, self.samples_in_datarecord(ii),
                                                 (<double*>db.data)+offset)
                print "readn this many samples", readn
                offset += self.samples_in_datarecord(ii)


###############################    
# low level functions



cdef unicode _ustring(s):
    if type(s) is unicode:
        # fast path for most common case(s)
        return <unicode>s
    elif PY_MAJOR_VERSION < 3 and isinstance(s, bytes):
        # only accept byte strings in Python 2.x, not in Py3
        return (<bytes>s).decode('ascii')
    elif isinstance(s, unicode):
        # an evil cast to <unicode> might work here in some(!) cases,
        # depending on what the further processing does.  to be safe,
        # we can always create a copy instead
        return unicode(s)
    else:
        raise TypeError()
        
# define a global name for whatever char type is used in the module
ctypedef unsigned char char_type

cdef char_type[:] _chars(s):
    if isinstance(s, unicode):
        # encode to the specific encoding used inside of the module
        s = (<unicode>s).encode('utf8')
    return s

def set_patientcode(int handle, char *patientcode):
    # check if rw?
    return c_edf.edf_set_patientcode(handle, patientcode)

cpdef int write_annotation_latin1(int handle, long long onset, long long duration, char *description):
        return c_edf.edfwrite_annotation_latin1(handle, onset, duration, description)

cpdef int write_annotation_utf8(int handle, long long onset, long long duration, char *description):
        return c_edf.edfwrite_annotation_utf8(handle, onset, duration, description)

cpdef int set_technician(int handle, char *technician):
    return c_edf.edf_set_technician(handle, technician)

cdef class EdfAnnotation:
    cdef c_edf.edf_annotation_struct annotation


cpdef int get_annotation(int handle, int n, EdfAnnotation edf_annotation):
    return c_edf.edf_get_annotation(handle, n, &(edf_annotation.annotation))

# need to use npbuffers
cpdef read_int_samples(int handle, int edfsignal, int n,
                         np.ndarray[np.int32_t,ndim=1] buf):
    """
/* reads n samples from edfsignal, starting from the current sample position indicator, into buf (edfsignal starts at 0) */
/* the values are the "raw" digital values */
/* bufsize should be equal to or bigger than sizeof(int[n]) */
/* the sample position indicator will be increased with the amount of samples read */
/* returns the amount of samples read (this can be less than n or zero!) */
/* or -1 in case of an error */


    ToDO!!!
    assert that these are stored as EDF/EDF+ files with int16 sized samples
    returns how many were actually read
    doesn't currently check that buf can hold all the data
    """
    return c_edf.edfread_digital_samples(handle, edfsignal, n,<int*>buf.data)

cpdef int blockwrite_digital_samples(int handle, np.ndarray[np.int16_t,ndim=1] buf):
    return c_edf.edf_blockwrite_digital_samples(handle, <int*>buf.data)

cpdef int blockwrite_physical_samples(int handle, np.ndarray[np.float64_t,ndim=1] buf):
    return c_edf.edf_blockwrite_physical_samples(handle, <double*>buf.data)

cpdef int set_recording_additional(int handle, char *recording_additional):
    return c_edf.edf_set_recording_additional(handle,recording_additional)

cpdef int write_physical_samples(int handle, np.ndarray[np.float64_t] buf):
    return c_edf.edfwrite_physical_samples(handle, <double *>buf.data)


cpdef int set_patientname(int handle, char *name):
    return c_edf.edf_set_patientname(handle, name)

cpdef int set_physical_minimum(int handle, int edfsignal, double phys_min):
    c_edf.edf_set_physical_minimum(handle, edfsignal, phys_min)

cpdef int read_physical_samples(int handle, int edfsignal, int n,
                                np.ndarray[np.float64_t] buf):
    return c_edf.edfread_physical_samples(handle, edfsignal, n, <double *>buf.data)

def close_file(handle):
    return c_edf.edfclose_file(handle)

# so you can use the same name if defining a python only function
def set_physical_maximum(handle, edfsignal, phys_max):
    return c_edf.edf_set_physical_maximum(handle, edfsignal, phys_max)

def open_file_writeonly(path, filetype, number_of_signals):
    """int edfopen_file_writeonly(char *path, int filetype, int number_of_signals)"""
    path_str = path.encode('UTF-8')
    return c_edf.edfopen_file_writeonly(path_str, filetype, number_of_signals)
    
def set_patient_additional(handle, patient_additional):
    """int edf_set_patient_additional(int handle, const char *patient_additional)"""
    return c_edf.edf_set_patient_additional(handle, patient_additional)

def set_digital_maximum(handle, edfsignal, dig_max):
    "int edf_set_digital_maximum(int handle, int edfsignal, int dig_max)"
    return c_edf.edf_set_digital_maximum(handle, edfsignal, dig_max)
        
# see CyEdfreader() class
# int edfopen_file_readonly(const char *path, struct edf_hdr_struct *edfhdr, int read_annotations)

def set_birthdate(handle, birthdate_year, birthdate_month, birthdate_day):
    """int edf_set_birthdate(int handle, int birthdate_year, int birthdate_month, int birthdate_day)"""
    return c_edf.edf_set_birthdate(handle, birthdate_year,  birthdate_month, birthdate_day)

def set_digital_minimum(handle, edfsignal, dig_min):
    """int edf_set_digital_minimum(int handle, int edfsignal, int dig_min)"""
    return c_edf.edf_set_digital_minimum(handle,  edfsignal, dig_min)

def write_digital_samples(handle, np.ndarray[np.int32_t] buf):
    """write_digital_samples(int handle, np.ndarray[np.int32_t] buf)"""
    return c_edf.edfwrite_digital_samples(handle, <int*>buf.data)

def set_equipment(handle, equipment):
    """int edf_set_equipment(int handle, const char *equipment)"""
    return c_edf.edf_set_equipment(handle, equipment)

def set_samplefrequency(handle, edfsignal, samplefrequency):
    """int edf_set_samplefrequency(int handle, int edfsignal, int samplefrequency)"""
    return c_edf.edf_set_samplefrequency(handle, edfsignal, samplefrequency)

def set_admincode(handle, admincode):
    """int edf_set_admincode(int handle, const char *admincode)"""
    return c_edf.edf_set_admincode(handle, admincode)

def set_label(handle, edfsignal, label):
    """int edf_set_label(int handle, int edfsignal, const char *label)"""
    return c_edf.edf_set_label(handle, edfsignal, label)


#FIXME need to make sure this gives the proper values for large values
def tell(handle, edfsignal):
    """long long edftell(int handle, int edfsignal)"""
    return c_edf.edftell(handle,  edfsignal)

def rewind(handle, edfsignal):
    """void edfrewind(int handle, int edfsignal)"""
    c_edf.edfrewind(handle, edfsignal)
    
def set_gender(handle, gender):
    """int edf_set_gender(int handle, int gender)"""
    return c_edf.edf_set_gender(handle, gender)

def set_physical_dimension(handle, edfsignal, phys_dim):
    """int edf_set_physical_dimension(int handle, int edfsignal, const char *phys_dim)"""
    return c_edf.edf_set_physical_dimension(handle, edfsignal, phys_dim)

def set_transducer(handle, edfsignal, transducer):
    """int edf_set_transducer(int handle, int edfsignal, const char *transducer)"""
    return c_edf.edf_set_transducer(handle, edfsignal, transducer)

def set_prefilter(handle, edfsignal, prefilter):
    """int edf_set_prefilter(int handle, int edfsignal, const char*prefilter)"""
    return c_edf.edf_set_prefilter(handle, edfsignal, prefilter)

def seek(handle, edfsignal, offset, whence):
    """long long edfseek(int handle, int edfsignal, long long offset, int whence)"""
    return c_edf.edfseek(handle, edfsignal, offset, whence)

def set_startdatetime(handle, startdate_year, startdate_month, startdate_day,
                                 starttime_hour, starttime_minute, starttime_second):
    """int edf_set_startdatetime(int handle, int startdate_year, int startdate_month, int startdate_day,
                                      int starttime_hour, int starttime_minute, int starttime_second)"""
    return c_edf.edf_set_startdatetime(handle, startdate_year, startdate_month, startdate_day,
                                 starttime_hour, starttime_minute, starttime_second)


def set_datarecord_duration(handle, duration):
    """int edf_set_datarecord_duration(int handle, int duration)"""
    return c_edf.edf_set_datarecord_duration(handle, duration)

