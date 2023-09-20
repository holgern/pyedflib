# Copyright (c) 2015-2023 Holger Nahrstaedt, Simon Kern
# Copyright (c) 2011, 2015, Chris Lee-Messer
# See LICENSE for license details.

__doc__ = """Cython wrapper for low-level C edflib implementation."""
__all__ = ['lib_version', 'CyEdfReader', 'set_patientcode', 'set_starttime_subsecond',
           'write_annotation_latin1', 'write_annotation_utf8', 'set_technician', 'EdfAnnotation',
           'get_annotation', 'read_int_samples', 'blockwrite_digital_samples', 'blockwrite_physical_samples',
           'set_recording_additional', 'write_physical_samples' ,'set_patientname', 'set_physical_minimum',
           'read_physical_samples', 'close_file', 'set_physical_maximum', 'open_file_writeonly',
           'set_patient_additional', 'set_digital_maximum', 'set_birthdate', 'set_digital_minimum',
           'write_digital_samples', 'set_equipment', 'set_samples_per_record','set_admincode', 'set_label',
           'tell', 'rewind', 'set_sex', 'set_gender', 'set_physical_dimension', 'set_transducer',
           'set_prefilter', 'seek', 'set_startdatetime' ,'set_datarecord_duration',
           'set_number_of_annotation_signals', 'open_errors', 'FILETYPE_EDFPLUS',
           'FILETYPE_EDF','FILETYPE_BDF','FILETYPE_BDFPLUS', 'write_errors', 'get_number_of_open_files',
           'get_handle', 'is_file_used', 'blockwrite_digital_short_samples', 'write_digital_short_samples']


#from c_edf cimport *
import locale
import os
import warnings
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
    EDFLIB_FILETYPE_ERROR              : "Wrong file type",
    EDFLIB_FILE_WRITE_ERROR            : "a write error occurred",
    EDFLIB_NUMBER_OF_SIGNALS_INVALID   : "The number of signals is invalid",
    EDFLIB_FILE_IS_DISCONTINUOUS       : "The file is discontinuous and cannot be read",
    EDFLIB_INVALID_READ_ANNOTS_VALUE   : "an annotation value could not be read",
    EDFLIB_FILE_ERRORS_STARTDATE      : "the file is not EDF(+) or BDF(+) compliant, the startdate is incorrect, it might contain incorrect characters, such as ':' instead of '.'",
    EDFLIB_FILE_ERRORS_STARTTIME      : "the file is not EDF(+) or BDF(+) compliant, the starttime is incorrect, it might contain incorrect characters, such as ':' instead of '.'",
    EDFLIB_FILE_ERRORS_NUMBER_SIGNALS : "the file is not EDF(+) or BDF(+) compliant (number of signals)",
    EDFLIB_FILE_ERRORS_BYTES_HEADER   : "the file is not EDF(+) or BDF(+) compliant (Bytes Header)",
    EDFLIB_FILE_ERRORS_RESERVED_FIELD : "the file is not EDF(+) or BDF(+) compliant (Reserved field)",
    EDFLIB_FILE_ERRORS_NUMBER_DATARECORDS : "the file is not EDF(+) or BDF(+) compliant (Number of Datarecords)",
    EDFLIB_FILE_ERRORS_DURATION : "the file is not EDF(+) or BDF(+) compliant (Duration)",
    EDFLIB_FILE_ERRORS_LABEL : "the file is not EDF(+) or BDF(+) compliant the label is incorrect",
    EDFLIB_FILE_ERRORS_TRANSDUCER : "the file is not EDF(+) or BDF(+) compliant the transducer is incorrect",
    EDFLIB_FILE_ERRORS_PHYS_DIMENSION : "the file is not EDF(+) or BDF(+) compliant (Physical Dimension)",
    EDFLIB_FILE_ERRORS_PHYS_MAX : "the file is not EDF(+) or BDF(+) compliant (Physical Maximum)",
    EDFLIB_FILE_ERRORS_PHYS_MIN : "the file is not EDF(+) or BDF(+) compliant (Physical Minimum)",
    EDFLIB_FILE_ERRORS_DIG_MAX : "the file is not EDF(+) or BDF(+) compliant (Digital Maximum)",
    EDFLIB_FILE_ERRORS_DIG_MIN : "the file is not EDF(+) or BDF(+) compliant (Digital Minimum)",
    EDFLIB_FILE_ERRORS_PREFILTER : "the file is not EDF(+) or BDF(+) compliant (Prefilter)",
    EDFLIB_FILE_ERRORS_SAMPLES_DATARECORD : "the file is not EDF(+) or BDF(+) compliant (Sample in Datarecord)",
    EDFLIB_FILE_ERRORS_FILESIZE : "the file is not EDF(+) or BDF(+) compliant (Filesize)",
    EDFLIB_FILE_ERRORS_RECORDINGFIELD : "the file is not EDF(+) or BDF(+) compliant (EDF+ Recordingfield)",
    EDFLIB_FILE_ERRORS_PATIENTNAME : "the file is not EDF(+) or BDF(+) compliant (EDF+ Patientname)",
    'default' : "unknown error"
    }

write_errors = {
    EDFLIB_MALLOC_ERROR                 : "malloc error",
    EDFLIB_NO_SUCH_FILE_OR_DIRECTORY    : "can not open file, no such file or directory",
    EDFLIB_MAXFILES_REACHED             : "to many files opened",
    EDFLIB_FILE_ALREADY_OPENED          : "file has already been opened",
    EDFLIB_FILETYPE_ERROR               : "Wrong file type",
    EDFLIB_FILE_WRITE_ERROR             : "a write error occurred",
    EDFLIB_NUMBER_OF_SIGNALS_INVALID    : "The number of signals is invalid",
    EDFLIB_NO_SIGNALS                   : "no signals to write",
    EDFLIB_TOO_MANY_SIGNALS             : "too many signals",
    EDFLIB_NO_SAMPLES_IN_RECORD         : "no samples in record",
    EDFLIB_DIGMIN_IS_DIGMAX             : "digmin is equal to digmax",
    EDFLIB_DIGMAX_LOWER_THAN_DIGMIN     : "digmax is lower than digmin",
    EDFLIB_PHYSMIN_IS_PHYSMAX           : "physmin is physmax",
    'default' : "unknown error"
}

# constants are redeclared here so we can access them from Python
FILETYPE_EDF = EDFLIB_FILETYPE_EDF
FILETYPE_EDFPLUS = EDFLIB_FILETYPE_EDFPLUS
FILETYPE_BDF = EDFLIB_FILETYPE_BDF
FILETYPE_BDFPLUS = EDFLIB_FILETYPE_BDFPLUS


def contains_unicode(string):
        try:
            string.encode('ascii')
            return False
        except:
            return True

def get_short_path_name(long_name):
    """
    Gets the short path name of a given long path.
    https://stackoverflow.com/a/23598461/200291
    """
    import ctypes
    from ctypes import wintypes
    _GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
    _GetShortPathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    _GetShortPathNameW.restype = wintypes.DWORD
    output_buf_size = 0
    while True:
        output_buf = ctypes.create_unicode_buffer(output_buf_size)
        needed = _GetShortPathNameW(long_name, output_buf, output_buf_size)
        if output_buf_size >= needed:
            return output_buf.value
        else:
            output_buf_size = needed

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


    cdef c_edf.edf_hdr_struct hdr
    cdef size_t nsamples_per_record
    #I think it is ok not to do this in __cinit__(*,**)
    def __init__(self, file_name, annotations_mode=EDFLIB_READ_ALL_ANNOTATIONS, check_file_size=EDFLIB_CHECK_FILE_SIZE):
        """
        EdfReader(file_name, annotations_mode, check_file_size)
        """
        self.hdr.handle = -1
        try:
            self.open(file_name, annotations_mode=annotations_mode, check_file_size=check_file_size)
        except FileNotFoundError as e:
            # if files contain Unicode on Windows, and the locale is set incorrectly
            # there can be errors when creating the file.
            # in this case, we can use a workaround and work on the file
            # using short file names (DOS style)
            exists = os.path.isfile(file_name)
            is_windows = os.name == 'nt'
            if exists and is_windows and contains_unicode(file_name):
                # work-around to at least make Unicode files readable at all
                warnings.warn('the filename {} contains Unicode, but Windows does not fully support this. ' \
                              'Please consider changing your locale to support UTF8. Attempting to '
                              'load file via workaround (https://github.com/holgern/pyedflib/pull/100) '.format(file_name))
                file_name = get_short_path_name(file_name)
                self.open(file_name, annotations_mode=annotations_mode, check_file_size=check_file_size)
            elif exists:
                raise OSError(123, 'File {} was found but can\'t be accessed. ' \
                              'Make sure it contains no special characters ' \
                              'or change your locale to use UTF8.'.format(file_name), None, 123)
            else:
                raise e


    def __dealloc__(self):
        if self.hdr.handle >= 0:
            c_edf.edfclose_file(self.hdr.handle)
            self.hdr.handle = -1

    def check_open_ok(self,result):
        if result == 0:
            return True
        else:
            msg = open_errors[self.hdr.filetype]
            if 'no such file or directory' in msg:
                raise FileNotFoundError, '{}: {}'.format(self.file_name, msg)
            raise OSError, '{}: {}'.format(self.file_name, msg)
            # return False

    def make_buffer(self):
        """
        utility function to make a buffer that can hold a single datarecord. This will
        hold the physical samples for a single data record as a numpy tensor.

        -  might extend to provide for N datarecord size

        """
        print ("self.hdr.datarecords_in_file", self.hdr.datarecords_in_file)
        tmp =0
        for ii in range(self.signals_in_file):
            tmp += self.samples_in_datarecord(ii)
        self.nsamples_per_record = tmp
        dbuffer = np.zeros(tmp, dtype='float64') # will get physical samples, not the original digital samples
        return dbuffer

    def open(self, file_name, annotations_mode=EDFLIB_READ_ALL_ANNOTATIONS, check_file_size=EDFLIB_CHECK_FILE_SIZE):
        """
        open(file_name, annotations_mode, check_file_size)
        """
        file_name_str = file_name.encode('utf_8','strict')
        result = c_edf.edfopen_file_readonly(file_name_str, &self.hdr, annotations_mode, check_file_size)

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
            return self.hdr.handle

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
            # duration is saved in resolution of 100 ns
            # therefore multiplying with EDFLIB_TIME_DIMENSION
            return self.hdr.file_duration/EDFLIB_TIME_DIMENSION

    property filetype:
        def __get__(self):
            return self.hdr.filetype

    property patient:
        "patient info (legacy EDF format)"
        def __get__(self):
            return self.hdr.patient.rstrip()

    property recording:
        "recording info (legacy EDF format)"
        def __get__(self):
            return self.hdr.recording.rstrip()

    property datarecord_duration:
        "datarecord duration in seconds (as a double)"
        def __get__(self):
            # duration is saved in resolution of 100 ns
            # therefore multiplying with EDFLIB_TIME_DIMENSION
            return (<double>self.hdr.datarecord_duration) / EDFLIB_TIME_DIMENSION

    property annotations_in_file:
        def __get__(self):
            return self.hdr.annotations_in_file

    property patientcode:
        def __get__(self):
            return self.hdr.patientcode

    property sex:
        def __get__(self):
            return self.hdr.gender

    property gender:
        def __get__(self):
            warnings.warn("Variable 'gender' is deprecated, use 'sex' instead.", DeprecationWarning, stacklevel=2)
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

    property starttime_subsecond:
        def __get__(self):
            return self.hdr.starttime_subsecond

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
        smp_per_record = <double>self.smp_per_record(channel)
        record_duration = self.datarecord_duration
        return smp_per_record / record_duration


    def smp_per_record(self, channel):
        return <int>self.hdr.signalparam[channel].smp_in_datarecord

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
        if self.hdr.handle >= 0:
            c_edf.edfclose_file(self.hdr.handle)
        self.hdr.handle = -1

    def read_digital_signal(self, signalnum, start, n, np.ndarray[np.int32_t, ndim=1] sigbuf):
        """
        read_digital_signal(self, signalnum, start, n, np.ndarray[np.int32_t, ndim=1] sigbuf
        read @n number of samples from signal number @signum starting at @start
        into numpy int32 array @sigbuf sigbuf must be at least n long
        """
        c_edf.edfseek(self.hdr.handle, signalnum, start, EDFSEEK_SET)
        readn = read_int_samples(self.hdr.handle, signalnum, n, sigbuf)
        if readn != n:
            print("read %d, less than %d requested!!!" % (readn, n))

    def readsignal(self, signalnum, start, n, np.ndarray[np.float64_t, ndim=1] sigbuf):

        """read @n number of samples from signal number @signum starting at
        @start into numpy float64 array @sigbuf sigbuf must be at least n long
        """

        c_edf.edfseek(self.hdr.handle, signalnum, start, EDFSEEK_SET)
        readn = c_edf.edfread_physical_samples(self.hdr.handle, signalnum, n, <double*>sigbuf.data)
        # print "read %d samples" % readn
        if readn != n:
            print ("read %d, less than %d requested!!!" % (readn, n))

    def load_datarecord(self, np.ndarray[np.float64_t, ndim=1] db, n=0):
        cdef size_t offset =0

        if n < self.hdr.datarecords_in_file:
            for ii in range(self.signals_in_file):
                c_edf.edfseek(self.hdr.handle, ii, n*self.samples_in_datarecord(ii), EDFSEEK_SET) # just a guess
                readn = c_edf.edfread_physical_samples(self.hdr.handle, ii, self.samples_in_datarecord(ii),
                                                 (<double*>db.data)+offset)
                print ("readn this many samples", readn)
                offset += self.samples_in_datarecord(ii)


###############################
# low level functions


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
    reads n samples from edfsignal, starting from the current sample position indicator, into buf (edfsignal starts at 0)
    the values are the "raw" digital values
    bufsize should be equal to or bigger than sizeof(int[n])
    the sample position indicator will be increased with the amount of samples read
    returns the amount of samples read (this can be less than n or zero!)
    or -1 in case of an error


    ToDO!!!
    assert that these are stored as EDF/EDF+ files with int16 sized samples
    returns how many were actually read
    doesn't currently check that buf can hold all the data
    """
    return c_edf.edfread_digital_samples(handle, edfsignal, n,<int*>buf.data)

cpdef int blockwrite_digital_samples(int handle, np.ndarray[np.int32_t,ndim=1] buf):
    return c_edf.edf_blockwrite_digital_samples(handle, <int*>buf.data)

cpdef int blockwrite_digital_short_samples(int handle, np.ndarray[np.int16_t,ndim=1] buf):
    return c_edf.edf_blockwrite_digital_short_samples(handle, <short*>buf.data)

cpdef int blockwrite_physical_samples(int handle, np.ndarray[np.float64_t,ndim=1] buf):
    return c_edf.edf_blockwrite_physical_samples(handle, <double*>buf.data)

cpdef int set_recording_additional(int handle, char *recording_additional):
    return c_edf.edf_set_recording_additional(handle,recording_additional)

cpdef int write_digital_short_samples(int handle, np.ndarray[np.int16_t] buf):
    return c_edf.edfwrite_digital_short_samples(handle, <short *>buf.data)

cpdef int write_physical_samples(int handle, np.ndarray[np.float64_t] buf):
    return c_edf.edfwrite_physical_samples(handle, <double *>buf.data)

cpdef int set_patientname(int handle, char *name):
    return c_edf.edf_set_patientname(handle, name)

cpdef int set_physical_minimum(int handle, int edfsignal, double phys_min):
    return c_edf.edf_set_physical_minimum(handle, edfsignal, phys_min)

cpdef int read_physical_samples(int handle, int edfsignal, int n,
                                np.ndarray[np.float64_t] buf):
    return c_edf.edfread_physical_samples(handle, edfsignal, n, <double *>buf.data)

def close_file(handle):
    return c_edf.edfclose_file(handle)

def get_number_of_open_files():
    return c_edf.edflib_get_number_of_open_files()

def get_handle(file_number):
    return c_edf.edflib_get_handle(file_number)

def is_file_used(path):
    path_byte = path.encode('utf_8','strict')
    return c_edf.edflib_is_file_used(path_byte)

# so you can use the same name if defining a python only function
def set_physical_maximum(handle, edfsignal, phys_max):
    return c_edf.edf_set_physical_maximum(handle, edfsignal, phys_max)

def open_file_writeonly(path, filetype, number_of_signals):
    """int edfopen_file_writeonly(char *path, int filetype, int number_of_signals)"""

    if os.name=='nt' and contains_unicode(path):
        default_enc = locale.getdefaultlocale()[1]
        if default_enc is None:
            default_enc = ''
        else:
            default_enc = default_enc.lower()
        using_unicode = 'utf' in default_enc or 'unicode' in default_enc or \
                        '10646' in  default_enc or default_enc=='cp65001'
        # Check if we're on Windows and the file path contains Unicode.
        # If so, use workaround to create file: In Python, create the file,
        # then look up and pass the short file name to the C library
        if not using_unicode:
            warnings.warn('Attempting to write Unicode file {} on Windows. ' \
                          'Consider changing your locale to UTF8.'.format(path))
            with open(path, 'wb'): pass
            path = get_short_path_name(path)

    py_byte_string  = path.encode('utf_8','strict')
    cdef char* path_str = py_byte_string
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

def set_samples_per_record(handle, edfsignal, smp_per_record ):
    """
    int set_samples_per_record(int handle, int edfsignal, int smp_per_record )

    sets how many samples are in the record for this signal.
    this is not the sampling frequency (Hz), (which is calculated by
    by smp_per_record/record_duration).
    """
    return c_edf.edf_set_samplefrequency(handle, edfsignal, smp_per_record)

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

def set_sex(handle, sex):
    """int edf_set_sex(int handle, int sex)"""
    if sex is None: return 0 #don't set sex at all to prevent default 'F'
    return c_edf.edf_set_gender(handle, sex)

def set_gender(handle, gender):
    warnings.warn("Function 'set_gender' is deprecated, use 'set_sex' instead.", DeprecationWarning, stacklevel=2)
    return set_sex(handle, gender)

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

def set_starttime_subsecond(handle, subsecond):
    """int edf_set_subsecond_starttime(int handle, int subsecond)"""
    return c_edf.edf_set_subsecond_starttime(handle, subsecond)

def set_datarecord_duration(handle, duration):
    """int edf_set_datarecord_duration(int handle, int duration)

    duration in seconds
    """
    # from the pyedflib documentation:
    # > To avoid rounding errors, the library stores some timevalues in variables
    # > of type long long int. In order not to loose the subsecond precision, all
    # > timevalues have been multiplied by 10000000. This will limit the
    # > timeresolution to 100 nanoSeconds. To calculate the amount of seconds,
    # > divide the timevalue by 10000000 or use the macro EDFLIB_TIME_DIMENSION
    # > which is declared in edflib.h.
    #
    # therefore, we divide by 100, and edflib internally multiplies by 100.
    # we could also change it in the C file, but better to leave that as is.
    duration *= EDFLIB_TIME_DIMENSION/100
    return c_edf.edf_set_datarecord_duration(handle, int(duration))

def set_number_of_annotation_signals(handle, annot_signals):
    """int edf_set_number_of_annotation_signals(int handle, int annot_signals)"""
    return c_edf.edf_set_number_of_annotation_signals(handle, annot_signals)
