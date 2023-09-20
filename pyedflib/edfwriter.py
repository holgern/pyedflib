# Copyright (c) 2019 - 2023 Simon Kern
# Copyright (c) 2015 - 2023 Holger Nahrstaedt
# Copyright (c) 2011, 2015, Chris Lee-Messer
# Copyright (c) 2016-2017 The pyedflib Developers
#                         <https://github.com/holgern/pyedflib>
# See LICENSE for license details.

import numpy as np
import sys
import warnings
from datetime import datetime, date
from ._extensions._pyedflib import FILETYPE_EDFPLUS, FILETYPE_BDFPLUS, FILETYPE_BDF, FILETYPE_EDF
from ._extensions._pyedflib import open_file_writeonly, set_physical_maximum, set_patient_additional, set_digital_maximum
from ._extensions._pyedflib import set_birthdate, set_digital_minimum, set_technician, set_recording_additional, set_patientname
from ._extensions._pyedflib import set_patientcode, set_equipment, set_admincode, set_sex, set_datarecord_duration, set_number_of_annotation_signals
from ._extensions._pyedflib import set_startdatetime, set_starttime_subsecond, set_samples_per_record, set_physical_minimum, set_label, set_physical_dimension
from ._extensions._pyedflib import set_transducer, set_prefilter, write_physical_samples, close_file, write_annotation_latin1, write_annotation_utf8
from ._extensions._pyedflib import blockwrite_physical_samples, write_errors, blockwrite_digital_samples, write_digital_short_samples, write_digital_samples, blockwrite_digital_short_samples


__all__ = ['EdfWriter']


def check_is_ascii(string):
    """according to the EDF+ specifications, only ASCII chars in ordeal
    range 32...126 are allowed, where 32 is space

    https://www.edfplus.info/specs/edfplus.html#header
    """
    if not all([ord(x)>32 and ord(x)<127 for x in string]):
        warnings.warn('Invalid char: header entries should contain only ASCII'\
                      ' characters and no spaces: "{}"'.format(string))


def check_signal_header_correct(channels, i, file_type):
    """
    helper function  to check if all entries in the channel dictionary are fine.

    Will give a warning if label, transducer, dimension, prefilter are too long.

    Will throw an exception if dmin, dmax, pmin, pmax are out of bounds or would
    be truncated in such a way as that signal values would be completely off.
    """
    ch = channels[i]
    label = ch['label']

    if len(ch['label'])>16:
        warnings.warn('Label of channel {} is longer than 16 ASCII chars.'\
                      'The label will be truncated to "{}"'.format(i, ch['label'][:16] ))
    if len(ch['prefilter'])>80:
        warnings.warn('prefilter of channel {} is longer than 80 ASCII chars.'\
                      'The label will be truncated to "{}"'.format(i, ch['prefilter'][:80] ))
    if len(ch['transducer'])>80:
        warnings.warn('transducer of channel {} is longer than 80 ASCII chars.'\
                      'The label will be truncated to "{}"'.format(i, ch['transducer'][:80] ))
    if len(ch['dimension'])>80:
        warnings.warn('dimension of channel {} is longer than 8 ASCII chars.'\
                      'The label will be truncated to "{}"'.format(i, ch['dimension'][:8] ))

    # these ones actually raise an exception
    dmin, dmax = (-8388608, 8388607) if file_type in (FILETYPE_BDFPLUS, FILETYPE_BDF) else (-32768, 32767)
    if ch['digital_min']<dmin:
        raise ValueError('Digital minimum for channel {} ({}) is {},'\
                         'but minimum allowed value is {}'.format(i, label,
                                                                  ch['digital_min'],
                                                                  dmin))
    if ch['digital_max']>dmax:
        raise ValueError('Digital maximum for channel {} ({}) is {},'\
                         'but maximum allowed value is {}'.format(i, label,
                                                                  ch['digital_max'],
                                                                  dmax))


    # if we truncate the physical min before the dot, we potentitally
    # have all the signals incorrect by an order of magnitude.
    if len(str(ch['physical_min']))>8 and ch['physical_min'] < -99999999:
        raise ValueError('Physical minimum for channel {} ({}) is {}, which has {} chars, '\
                         'however, EDF+ can only save 8 chars, critical precision loss is expected, '\
                         'please convert the signals to another dimesion (eg uV to mV)'.format(i, label,
                                                                      ch['physical_min'],
                                                                      len(str(ch['physical_min']))))
    if len(str(ch['physical_max']))>8 and ch['physical_max'] > 99999999:
        raise ValueError('Physical minimum for channel {} ({}) is {}, which has {} chars, '\
                         'however, EDF+ can only save 8 chars, critical precision loss is expected, '\
                         'please convert the signals to another dimesion (eg uV to mV).'.format(i, label,
                                                                      ch['physical_max'],
                                                                      len(str(ch['physical_max']))))
    # if we truncate the physical min behind the dot, we just lose precision,
    # in this case only a warning is enough
    if len(str(ch['physical_min']))>8:
        warnings.warn('Physical minimum for channel {} ({}) is {}, which has {} chars, '\
                         'however, EDF+ can only save 8 chars, will be truncated to {}, '\
                         'some loss of precision is to be expected'.format(i, label,
                                                                      ch['physical_min'],
                                                                      len(str(ch['physical_min'])),
                                                                      str(ch['physical_min'])[:8]))
    if len(str(ch['physical_max']))>8:
        warnings.warn('Physical minimum for channel {} ({}) is {}, which has {} chars, '\
                         'however, EDF+ can only save 8 chars, will be truncated to {}, '\
                         'some loss of precision is to be expected.'.format(i, label,
                                                                      ch['physical_max'],
                                                                      len(str(ch['physical_max'])),
                                                                      str(ch['physical_max'])[:8]))



def u(x):
    return x.decode("utf_8", "strict")


def du(x):
    if isinstance(x, bytes):
        return x
    else:
        return x.encode("utf_8")


def isstr(s):
    warnings.warn("Function 'isstr' is deprecated.", DeprecationWarning, stacklevel=2)
    return isinstance(s, str)


def isbytestr(s):
    warnings.warn("Function 'isbytestr' is deprecated.", DeprecationWarning, stacklevel=2)
    return isinstance(s, bytes)


def sex2int(sex):
    if isinstance(sex, int) or sex is None:
        return sex
    elif sex.lower() in ('', 'x', 'xx', 'xxx', 'unknown', '?', '??'):
        return None
    elif sex.lower() in ("female", "woman", "f", "w"):
        return 0
    elif sex.lower() in  ("male", "man", "m"):
        return 1
    else:
        raise ValueError(f"Unknown sex: '{sex}'")


def gender2int(gender):
    warnings.warn("Function 'gender2int' is deprecated, use 'sex2int' instead.", DeprecationWarning, stacklevel=2)
    return sex2int(gender)


class ChannelDoesNotExist(Exception):
    def __init__(self, value):
        self.parameter = value

    def __str__(self):
        return repr(self.parameter)


class WrongInputSize(Exception):
    def __init__(self, value):
        self.parameter = value

    def __str__(self):
        return repr(self.parameter)


class EdfWriter:
    def __exit__(self, exc_type, exc_val, ex_tb):
        self.close()

    def __enter__(self):
        return self

    def __del__(self):
        self.close()

    def __init__(self, file_name, n_channels,
                 file_type=FILETYPE_EDFPLUS):
        """Initialises an EDF file at file_name.
        file_type is one of
            edflib.FILETYPE_EDFPLUS
            edflib.FILETYPE_BDFPLUS
        n_channels is the number of channels without the annotation channel

        channel_info should be a
        list of dicts, one for each channel in the data. Each dict needs
        these values:

            'label' : channel label (string, <= 16 characters, must be unique)
            'dimension' : physical dimension (e.g., mV) (string, <= 8 characters)
            'sample_rate' : sample frequency in hertz (int). Deprecated: use 'sample_frequency' instead.
            'sample_frequency' : number of samples per record (int)
            'physical_max' : maximum physical value (float)
            'physical_min' : minimum physical value (float)
            'digital_max' : maximum digital value (int, -2**15 <= x < 2**15)
            'digital_min' : minimum digital value (int, -2**15 <= x < 2**15)
        """
        self.path = file_name
        self.file_type = file_type
        self.patient_name = ''
        self.patient_code = ''
        self.technician = ''
        self.equipment = ''
        self.recording_additional = ''
        self.patient_additional = ''
        self.admincode = ''
        self.sex = None
        self.recording_start_time = datetime.now().replace(microsecond=0)

        self.birthdate = ''
        self.record_duration = 1  # length of one data record in seconds
        self.number_of_annotations = 1 if file_type in [FILETYPE_EDFPLUS, FILETYPE_BDFPLUS] else 0
        self.n_channels = n_channels
        self.channels = []
        self.sample_buffer = []
        for i in np.arange(self.n_channels):
            if self.file_type == FILETYPE_BDFPLUS or self.file_type == FILETYPE_BDF:
                self.channels.append({'label': f'ch{i}', 'dimension': 'mV', 'sample_rate': 100,
                                      'sample_frequency': None, 'physical_max': 1.0, 'physical_min': -1.0,
                                      'digital_max': 8388607,'digital_min': -8388608,
                                      'prefilter': '', 'transducer': ''})
            elif self.file_type == FILETYPE_EDFPLUS or self.file_type == FILETYPE_EDF:
                self.channels.append({'label': f'ch{i}', 'dimension': 'mV', 'sample_rate': 100,
                                      'sample_frequency': None, 'physical_max': 1.0, 'physical_min': -1.0,
                                      'digital_max': 32767, 'digital_min': -32768,
                                      'prefilter': '', 'transducer': ''})

                self.sample_buffer.append([])
        self.handle = open_file_writeonly(self.path, self.file_type, self.n_channels)
        if (self.handle < 0):
            raise OSError(write_errors[self.handle])
        self._enforce_record_duration = False

    def update_header(self):
        """
        Updates header to edffile struct
        """
        # some checks that warn users if header fields exceed 80 chars
        patient_ident = len(self.patient_code) + len(self.patient_name) \
                        + len(self.patient_additional) + 3 + 1 + 11 # 3 spaces 1 sex 11 birthdate
        record_ident = len(self.equipment) + len(self.technician) \
                       + len(self.admincode) + len(self.recording_additional) \
                       + len('Startdate') + 3 + 11 # 3 spaces 11 birthdate

        if patient_ident>80:
            warnings.warn('Patient code, name, sex and birthdate combined must not be larger than 80 chars. ' +
                          f'Currently has len of {patient_ident}. See https://www.edfplus.info/specs/edfplus.html#additionalspecs')
        if record_ident>80:
            warnings.warn('Equipment, technician, admincode and recording_additional combined must not be larger than 80 chars. ' +
                          f'Currently has len of {record_ident}. See https://www.edfplus.info/specs/edfplus.html#additionalspecs')

        # all data records (i.e. blocks of data of a channel) have one singular
        # length in seconds. If there are different sampling frequencies for
        # the channels, we need to find a common denominator so that all sample
        # frequencies can be represented accurately.
        # this can be overwritten by explicitly calling setDatarecordDuration
        self._calculate_optimal_record_duration()

        set_technician(self.handle, du(self.technician))
        set_recording_additional(self.handle, du(self.recording_additional))
        set_patientname(self.handle, du(self.patient_name))
        set_patientcode(self.handle, du(self.patient_code))
        set_patient_additional(self.handle, du(self.patient_additional))
        set_equipment(self.handle, du(self.equipment))
        set_admincode(self.handle, du(self.admincode))
        set_sex(self.handle, sex2int(self.sex))

        set_datarecord_duration(self.handle, self.record_duration)
        set_number_of_annotation_signals(self.handle, self.number_of_annotations)
        set_startdatetime(self.handle, self.recording_start_time.year, self.recording_start_time.month,
                          self.recording_start_time.day, self.recording_start_time.hour,
                          self.recording_start_time.minute, self.recording_start_time.second)
        # subseconds are noted in nanoseconds, so we multiply by 100
        if self.recording_start_time.microsecond>0:
            set_starttime_subsecond(self.handle, self.recording_start_time.microsecond*100)
        if isinstance(self.birthdate, str):
            if self.birthdate != '':
                birthday = datetime.strptime(self.birthdate, '%d %b %Y').date()
                set_birthdate(self.handle, birthday.year, birthday.month, birthday.day)
        else:
            set_birthdate(self.handle, self.birthdate.year, self.birthdate.month, self.birthdate.day)

        for i in np.arange(self.n_channels):
            check_signal_header_correct(self.channels, i, self.file_type)
            set_samples_per_record(self.handle, i, self.get_smp_per_record(i))
            set_physical_maximum(self.handle, i, self.channels[i]['physical_max'])
            set_physical_minimum(self.handle, i, self.channels[i]['physical_min'])
            set_digital_maximum(self.handle, i, self.channels[i]['digital_max'])
            set_digital_minimum(self.handle, i, self.channels[i]['digital_min'])
            set_label(self.handle, i, du(self.channels[i]['label']))
            set_physical_dimension(self.handle, i, du(self.channels[i]['dimension']))
            set_transducer(self.handle, i, du(self.channels[i]['transducer']))
            set_prefilter(self.handle, i, du(self.channels[i]['prefilter']))

    def setHeader(self, fileHeader):
        """
        Sets the file header
        """
        self.technician = fileHeader["technician"]
        self.recording_additional = fileHeader["recording_additional"]
        self.patient_name = fileHeader["patientname"]
        self.patient_additional = fileHeader["patient_additional"]
        self.patient_code = fileHeader["patientcode"]
        self.equipment = fileHeader["equipment"]
        self.admincode = fileHeader["admincode"]
        self.sex = fileHeader["sex"]
        self.recording_start_time = fileHeader["startdate"]
        self.birthdate = fileHeader["birthdate"]
        self.update_header()

    def setSignalHeader(self, edfsignal, channel_info):
        """
        Sets the parameter for signal edfsignal.

        channel_info should be a dict with
        these values:

            'label' : channel label (string, <= 16 characters, must be unique)
            'dimension' : physical dimension (e.g., mV) (string, <= 8 characters)
            'sample_rate' : sample frequency in hertz (int). Deprecated: use 'sample_frequency' instead.
            'sample_frequency' : number of samples per record (int)
            'physical_max' : maximum physical value (float)
            'physical_min' : minimum physical value (float)
            'digital_max' : maximum digital value (int, -2**15 <= x < 2**15)
            'digital_min' : minimum digital value (int, -2**15 <= x < 2**15)
        """
        if edfsignal < 0 or edfsignal > self.n_channels:
            raise ChannelDoesNotExist(edfsignal)
        self.channels[edfsignal].update(channel_info)
        self.update_header()

    def setSignalHeaders(self, signalHeaders):
        """
        Sets the parameter for all signals

        Parameters
        ----------
        signalHeaders : array_like
            containing dict with
                'label' : str
                          channel label (string, <= 16 characters, must be unique)
                'dimension' : str
                          physical dimension (e.g., mV) (string, <= 8 characters)
                'sample_rate' :
                          sample frequency in hertz (int). Deprecated: use 'sample_frequency' instead.
                'sample_frequency' : int
                          number of samples per record
                'physical_max' : float
                          maximum physical value
                'physical_min' : float
                         minimum physical value
                'digital_max' : int
                         maximum digital value (-2**15 <= x < 2**15)
                'digital_min' : int
                         minimum digital value (-2**15 <= x < 2**15)
        """
        for edfsignal in np.arange(self.n_channels):
            self.channels[edfsignal].update(signalHeaders[edfsignal])
        self.update_header()

    def setTechnician(self, technician):
        """
        Sets the technicians name to `technician`.

        Notes
        -----
        This function is optional and can be called only after opening a file in writemode and before the first sample write action.
        """
        check_is_ascii(technician)
        self.technician = technician
        self.update_header()

    def setRecordingAdditional(self, recording_additional):
        """
        Sets the additional recordinginfo

        Notes
        -----
        This function is optional and can be called only after opening a file in writemode and before the first sample write action.
        """
        check_is_ascii(recording_additional)
        self.recording_additional = recording_additional
        self.update_header()

    def setPatientName(self, patient_name):
        """
        Sets the patientname to `patient_name`.

        Notes
        -----
        This function is optional and can be called only after opening a file in writemode and before the first sample write action.
        """
        check_is_ascii(patient_name)
        self.patient_name = patient_name
        self.update_header()

    def setPatientCode(self, patient_code):
        """
        Sets the patientcode to `patient_code`.

        Notes
        -----
        This function is optional and can be called only after opening a file in writemode and before the first sample write action.
        """
        check_is_ascii(patient_code)
        self.patient_code = patient_code
        self.update_header()

    def setPatientAdditional(self, patient_additional):
        """
        Sets the additional patientinfo to `patient_additional`.

        Notes
        -----
        This function is optional and can be called only after opening a file in writemode and before the first sample write action.
        """
        check_is_ascii(patient_additional)
        self.patient_additional = patient_additional
        self.update_header()

    def setEquipment(self, equipment):
        """
        Sets the name of the param equipment used during the acquisition.
        This function is optional and can be called only after opening a file in writemode and before the first sample write action.

        Parameters
        ----------
        equipment : str
            Describes the measurement equpipment

        """
        check_is_ascii(equipment)
        self.equipment = equipment
        self.update_header()

    def setAdmincode(self, admincode):
        """
        Sets the admincode.

        This function is optional and can be called only after opening a file in writemode and before the first sample write action.

        Parameters
        ----------
        admincode : str
            admincode which is written into the header

        """
        check_is_ascii(admincode)
        self.admincode = admincode
        self.update_header()

    def setSex(self, sex):
        """
        Sets the sex. Due to the edf specifications, only binary assignment is possible.
        This function is optional and can be called only after opening a file in writemode and before the first sample write action.

        Parameters
        ----------
        sex : int
            1 is male, 0 is female
        """
        self.sex = sex2int(sex)
        self.update_header()

    def setGender(self, gender):
        warnings.warn("Function 'setGender' is deprecated, use 'setSex' instead.", DeprecationWarning, stacklevel=2)
        self.sex = sex2int(gender)
        self.update_header()

    def setDatarecordDuration(self, record_duration):
        """
        Sets the datarecord duration. The default value is 1 second.
        The datarecord duration must be in the range 0.001 to 60  seconds.
        Usually, the datarecord duration is calculated automatically to
        ensure that all sample frequencies are representable, nevertheless,
        you can overwrite the datarecord duration manually. This can, however,
        lead to unexpected side-effects in the sample frequency calculations.

        Returns 0 on success, otherwise -1.

        Parameters
        ----------
        record_duration : float
            Sets the datarecord duration in units of seconds

        Notes
        -----
        This function is NOT REQUIRED but can be called after opening a file
        in writemode and before the first sample write action. This function
        can be used when you want to use a samplefrequency which is not an
        integer. For example, if you want to use a samplerate of 0.5 Hz, set
        the samplefrequency to 5 Hz and the datarecord duration to 10 seconds.
        Do not use this function, except when absolutely necessary!
        """
        self._enforce_record_duration = True
        self.record_duration = record_duration
        self.update_header()

    def set_number_of_annotation_signals(self, number_of_annotations):
        """
        Sets the number of annotation signals. The default value is 1
        This function is optional and can be called only after opening a file in writemode
        and before the first sample write action
        Normally you don't need to change the default value. Only when the number of annotations
        you want to write is more than the number of seconds of the duration of the recording, you can use
        this function to increase the storage space for annotations
        Minimum is 1, maximum is 64

        Parameters
        ----------
        number_of_annotations : integer
            Sets the number of annotation signals
        """
        number_of_annotations = max((min((int(number_of_annotations), 64)), 1))
        self.number_of_annotations = number_of_annotations
        self.update_header()

    def setStartdatetime(self, recording_start_time):
        """
        Sets the recording start Time

        Parameters
        ----------
        recording_start_time: datetime object
            Sets the recording start Time
        """
        if not isinstance(recording_start_time, datetime):
            recording_start_time = datetime.strptime(recording_start_time,"%d %b %Y %H:%M:%S")
        self.recording_start_time = recording_start_time
        self.update_header()

    def setBirthdate(self, birthdate):
        """
        Sets the birthdate.

        Parameters
        ----------
        birthdate: date object from datetime

        Examples
        --------
        >>> import pyedflib
        >>> from datetime import datetime, date
        >>> f = pyedflib.EdfWriter('test.bdf', 1, file_type=pyedflib.FILETYPE_BDFPLUS)
        >>> f.setBirthdate(date(1951, 8, 2))
        >>> f.close()

        Notes
        -----
        This function is optional and can be called only after opening a file in writemode and before the first sample write action.
        """
        if isinstance(birthdate, str):
            birthdate = datetime.strptime(birthdate, "%d.%m.%Y")
        self.birthdate = birthdate
        self.update_header()

    def setSamplefrequency(self, edfsignal, samplefrequency):
        """
        Sets the samplefrequency of signal edfsignal.

        Parameters
        ----------
        edfsignal: index number of signal
        samplefrequency: int or float stating the sampling frequency in Hz.
                        internally, EDF stores the sampling frequency by
                        setting the smp_per_record and the record_duration.
                        That means, the sampling frequency is not stored
                        explicitly.

        Notes
        -----
        This function is required for every signal and can be called only after
        opening a file in writemode and before the first sample write action.
        """
        if edfsignal < 0 or edfsignal > self.n_channels:
            raise ChannelDoesNotExist(edfsignal)

        self.channels[edfsignal]['sample_frequency'] = samplefrequency
        self.update_header()

    def setPhysicalMaximum(self, edfsignal, physical_maximum):
        """
        Sets the physical_maximum of signal edfsignal.

        Parameters
        ----------
        edfsignal: int
            signal number
        physical_maximum: float
            Sets the physical maximum

        Notes
        -----
        This function is required for every signal and can be called only after opening a file in writemode and before the first sample write action.
        """
        if edfsignal < 0 or edfsignal > self.n_channels:
            raise ChannelDoesNotExist(edfsignal)
        self.channels[edfsignal]['physical_max'] = physical_maximum
        self.update_header()

    def setPhysicalMinimum(self, edfsignal, physical_minimum):
        """
        Sets the physical_minimum of signal edfsignal.

        Parameters
        ----------
        edfsignal: int
            signal number
        physical_minimum: float
            Sets the physical minimum

        Notes
        -----
        This function is required for every signal and can be called only after opening a file in writemode and before the first sample write action.
        """
        if (edfsignal < 0 or edfsignal > self.n_channels):
            raise ChannelDoesNotExist(edfsignal)
        self.channels[edfsignal]['physical_min'] = physical_minimum
        self.update_header()

    def setDigitalMaximum(self, edfsignal, digital_maximum):
        """
        Sets the maximum digital value of signal edfsignal.
        Usually, the value 32767 is used for EDF+ and 8388607 for BDF+.

        Parameters
        ----------
        edfsignal : int
            signal number
        digital_maximum : int
            Sets the maximum digital value

        Notes
        -----
        This function is optional and can be called only after opening a file in writemode and before the first sample write action.
        """
        if (edfsignal < 0 or edfsignal > self.n_channels):
            raise ChannelDoesNotExist(edfsignal)
        self.channels[edfsignal]['digital_max'] = digital_maximum
        self.update_header()

    def setDigitalMinimum(self, edfsignal, digital_minimum):
        """
        Sets the minimum digital value of signal edfsignal.
        Usually, the value -32768 is used for EDF+ and -8388608 for BDF+. Usually this will be (-(digital_maximum + 1)).

        Parameters
        ----------
        edfsignal : int
            signal number
        digital_minimum : int
            Sets the minimum digital value

        Notes
        -----
        This function is optional and can be called only after opening a file in writemode and before the first sample write action.
        """
        if (edfsignal < 0 or edfsignal > self.n_channels):
            raise ChannelDoesNotExist(edfsignal)
        self.channels[edfsignal]['digital_min'] = digital_minimum
        self.update_header()

    def setLabel(self, edfsignal, label):
        """
        Sets the label (name) of signal edfsignal ("FP1", "SaO2", etc.).

        Parameters
        ----------
        edfsignal : int
            signal number on which the label should be changed
        label : str
            signal label

        Notes
        -----
        This function is recommended for every signal and can be called only after opening a file in writemode and before the first sample write action.
        """
        if (edfsignal < 0 or edfsignal > self.n_channels):
            raise ChannelDoesNotExist(edfsignal)
        self.channels[edfsignal]['label'] = label
        self.update_header()

    def setPhysicalDimension(self, edfsignal, physical_dimension):
        """
        Sets the physical dimension of signal edfsignal ("uV", "BPM", "mA", "Degr.", etc.)

        :param edfsignal: int
        :param physical_dimension: str

        Notes
        -----
        This function is recommended for every signal and can be called only after opening a file in writemode and before the first sample write action.
        """
        if edfsignal < 0 or edfsignal > self.n_channels:
            raise ChannelDoesNotExist(edfsignal)
        self.channels[edfsignal]['dimension'] = physical_dimension
        self.update_header()

    def setTransducer(self, edfsignal, transducer):
        """
        Sets the transducer of signal edfsignal

        :param edfsignal: int
        :param transducer: str

        Notes
        -----
        This function is optional for every signal and can be called only after opening a file in writemode and before the first sample write action.
        """
        if (edfsignal < 0 or edfsignal > self.n_channels):
            raise ChannelDoesNotExist(edfsignal)
        self.channels[edfsignal]['transducer'] = transducer
        self.update_header()

    def setPrefilter(self, edfsignal, prefilter):
        """
        Sets the prefilter of signal edfsignal ("HP:0.1Hz", "LP:75Hz N:50Hz", etc.)

        :param edfsignal: int
        :param prefilter: str

        Notes
        -----
        This function is optional for every signal and can be called only after opening a file in writemode and before the first sample write action.
        """
        if edfsignal < 0 or edfsignal > self.n_channels:
            raise ChannelDoesNotExist(edfsignal)
        self.channels[edfsignal]['prefilter'] = prefilter
        self.update_header()

    def writePhysicalSamples(self, data):
        """
        Writes n physical samples (uV, mA, Ohm) belonging to one signal where n
        is the samplefrequency of the signal.

        data_vec belonging to one signal. The size must be the samplefrequency of the signal.

        Notes
        -----
        Writes n physical samples (uV, mA, Ohm) from data_vec belonging to one signal where n
        is the samplefrequency of the signal. The physical samples will be converted to digital
        samples using the values of physical maximum, physical minimum, digital maximum and digital
        minimum. The number of samples written is equal to the samplefrequency of the signal.
        Call this function for every signal in the file. The order is important! When there are 4
        signals in the file, the order of calling this function must be: signal 0, signal 1, signal 2,
        signal 3, signal 0, signal 1, signal 2, etc.

        All parameters must be already written into the bdf/edf-file.
        """
        return write_physical_samples(self.handle, data)

    def writeDigitalSamples(self, data):
        return write_digital_samples(self.handle, data)

    def writeDigitalShortSamples(self, data):
        return write_digital_short_samples(self.handle, data)

    def blockWritePhysicalSamples(self, data):
        """
        Writes physical samples (uV, mA, Ohm)
        must be filled with samples from all signals
        where each signal has n samples which is the samplefrequency of the signal.

        data_vec belonging to one signal. The size must be the samplefrequency of the signal.

        Notes
        -----
        buf must be filled with samples from all signals, starting with signal 0, 1, 2, etc.
        one block equals one second
        The physical samples will be converted to digital samples using the
        values of physical maximum, physical minimum, digital maximum and digital minimum
        The number of samples written is equal to the sum of the samplefrequencies of all signals
        Size of buf should be equal to or bigger than sizeof(double) multiplied by the sum of the samplefrequencies of all signals
        Returns 0 on success, otherwise -1

        All parameters must be already written into the bdf/edf-file.
        """
        return blockwrite_physical_samples(self.handle, data)

    def blockWriteDigitalSamples(self, data):
        return blockwrite_digital_samples(self.handle, data)

    def blockWriteDigitalShortSamples(self, data):
        return blockwrite_digital_short_samples(self.handle, data)

    def writeSamples(self, data_list, digital = False):
        """
        Writes physical samples (uV, mA, Ohm) from data belonging to all signals
        The physical samples will be converted to digital samples using the values
        of physical maximum, physical minimum, digital maximum and digital minimum.
        if the samplefrequency of all signals are equal, then the data could be
        saved into a matrix with the size (N,signals) If the samplefrequency
        is different, then sample_freq is a vector containing all the different
        samplefrequencys. The data is saved as list. Each list entry contains
        a vector with the data of one signal.

        If digital is True, digital signals (as directly from the ADC) will be expected.
        (e.g. int16 from 0 to 2048)

        All parameters must be already written into the bdf/edf-file.
        """
        there_are_blank_sample_frequencies = any([channel.get('sample_frequency') is None
                                                 for channel in self.channels])
        if there_are_blank_sample_frequencies:
            warnings.warn("The 'sample_rate' parameter is deprecated. Please use "
                          "'sample_frequency' instead.", DeprecationWarning)

        if (len(data_list)) == 0:
            raise WrongInputSize('Data list is empty')
        if (len(data_list) != len(self.channels)):
            raise WrongInputSize('Number of channels ({}) \
             unequal to length of data ({})'.format(len(self.channels), len(data_list)))

        # Check for F-contiguous arrays
        if not all(s.flags.c_contiguous for s in data_list):
            warnings.warn('signals are in Fortran order. Will automatically '
                          'transfer to C order for compatibility with edflib.')
            data_list = np.ascontiguousarray(data_list)

        if digital:
            if any([not np.issubdtype(a.dtype, np.integer) for a in data_list]):
                raise TypeError('Digital = True requires all signals in int')

        # Check that all channels have different physical_minimum and physical_maximum
        for chan in self.channels:
            assert chan['physical_min'] != chan['physical_max'], \
            'In chan {} physical_min {} should be different from '\
            'physical_max {}'.format(chan['label'], chan['physical_min'], chan['physical_max'])

        ind = []
        notAtEnd = True
        for i in np.arange(len(data_list)):
            ind.append(0)

        sampleLength = 0
        smp_per_record = np.zeros(len(data_list), dtype=np.int32)
        for i in np.arange(len(data_list)):
            smp_per_record[i] = self.get_smp_per_record(i)

            if (np.size(data_list[i]) < ind[i] + smp_per_record[i]):
                notAtEnd = False
            sampleLength += smp_per_record[i]

        dataRecord = np.array([], dtype=np.int32 if digital else None)

        while notAtEnd:
            del dataRecord
            dataRecord = np.array([], dtype=np.int32 if digital else None)
            for i in np.arange(len(data_list)):
                dataRecord = np.append(dataRecord, data_list[i][int(ind[i]):int(ind[i]+smp_per_record[i])])
                ind[i] += smp_per_record[i]
            if digital:
                success = self.blockWriteDigitalSamples(dataRecord)
            else:
                success = self.blockWritePhysicalSamples(dataRecord)

            if success < 0:
                raise OSError(f'Unknown error while calling blockWriteSamples: {success}')

            for i in np.arange(len(data_list)):
                if (np.size(data_list[i]) < ind[i] + smp_per_record[i]):
                    notAtEnd = False


        for i in np.arange(len(data_list)):
            lastSamples = np.zeros(smp_per_record[i], dtype=np.int32 if digital else None)
            lastSampleInd = int(np.max(data_list[i].shape) - ind[i])
            lastSampleInd = int(np.min((lastSampleInd, smp_per_record[i])))
            if lastSampleInd > 0:
                lastSamples[:lastSampleInd] = data_list[i][-lastSampleInd:]
                if digital:
                    success = self.writeDigitalSamples(lastSamples)
                else:
                    success = self.writePhysicalSamples(lastSamples)

                if success<0:
                    raise OSError(f'Unknown error while calling writeSamples: {success}')

    def writeAnnotation(self, onset_in_seconds, duration_in_seconds, description, str_format='utf_8'):
        """
        Writes an annotation/event to the file
        """
        if self.file_type in [FILETYPE_EDF, FILETYPE_BDF]:
            raise TypeError('Trying to write annotation to EDF/BDF, must use EDF+/BDF+')

        if isinstance(duration_in_seconds, bytes):
            duration_in_seconds = float(duration_in_seconds)

        if str_format == 'utf_8':
            if duration_in_seconds >= 0:
                return write_annotation_utf8(self.handle, np.round(onset_in_seconds*10000).astype(np.int64), np.round(duration_in_seconds*10000).astype(int), du(description))
            else:
                return write_annotation_utf8(self.handle, np.round(onset_in_seconds*10000).astype(np.int64), -1, du(description))
        else:
            if duration_in_seconds >= 0:
                return write_annotation_latin1(self.handle, np.round(onset_in_seconds*10000).astype(np.int64), np.round(duration_in_seconds*10000).astype(int), u(description).encode('latin1'))
            else:
                return write_annotation_latin1(self.handle, np.round(onset_in_seconds*10000).astype(np.int64), -1, u(description).encode('latin1'))

    def close(self):
        """
        Closes the file.
        """
        close_file(self.handle)
        self.handle = -1

    def get_smp_per_record(self, ch_idx):
        """
        gets the calculated number of samples that need to be fit into one
        record (i.e. window/block of data) with the given record duration.
        """
        fs = self._get_sample_frequency(ch_idx)
        if fs is None: return None

        record_duration = self.record_duration
        smp_per_record = fs*record_duration

        if not np.isclose(np.round(smp_per_record), np.round(smp_per_record, 6)):
            warnings.warn(f'Sample frequency {fs} can not be represented accurately. \n' +
                          f'smp_per_record={smp_per_record}, record_duration={record_duration} seconds,' +
                          f'calculated sample_frequency will be {np.round(smp_per_record)/record_duration}')
        return int(np.round(smp_per_record))


    def _calculate_optimal_record_duration(self):
        """
        calculate optimal denominator (record duration in seconds)
        for all sample frequencies such that smp_per_record is an integer
        for all channels.

        If all sampling frequencies are integers, this will simply be 1.
        """
        if self._enforce_record_duration: return
        allint = lambda int_list: all([n==int(n) for n in int_list])
        all_fs = [self._get_sample_frequency(i) for i,_ in enumerate(self.channels)]

        # calculate the optimal record duration to represent all frequencies.
        # this is achieved when fs*duration=int, i.e. the number of samples
        # in one data record can be represented by an int (smp_per_record)
        # if all sampling frequencies are ints, this will be simply 1
        # for now this brute force solution should cover 99% of cases.
        # TODO: optimize this process

        record_duration = 0
        for i in range(1, 60):
            if allint([x*i for x in all_fs]):
                record_duration = i
                break
        assert record_duration>0, f'cannot accurately represent sampling frequencies with data record durations between 1-60s: {all_fs}'
        assert record_duration<=60, 'record duration must be below 60 seconds'
        self.record_duration = record_duration

    def _get_sample_frequency(self, channelIndex):
        # Temporary conditional assignment while we deprecate 'sample_rate' as a channel attribute
        # in favor of 'sample_frequency', supporting the use of either to give
        # users time to switch to the new interface.
        if 'sample_rate' in self.channels[channelIndex]:
            warnings.warn("`sample_rate` is deprecated and will be removed in a future release. \
                          Please use `sample_frequency` instead", DeprecationWarning)
        return (self.channels[channelIndex]['sample_rate']
                if self.channels[channelIndex].get('sample_frequency') is None
                else self.channels[channelIndex]['sample_frequency'])
