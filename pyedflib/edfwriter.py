# -*- coding: utf-8 -*-
# Copyright (c) 2015 Chris Lee-Messer <https://bitbucket.org/cleemesser/>
# Copyright (c) 2015 Holger Nahrstaedt

from __future__ import division, print_function, absolute_import

__all__ = ['EdfWriter']

import numpy as np
from datetime import datetime, date

from ._edflib import *


class EdfWriter(object):
    def __exit__(self, exc_type, exc_val, ex_tb):
        self.close()  # cleanup the file

    def __init__(self, file_name, n_channels,
                 file_type=FILETYPE_EDFPLUS):
        '''Initialises an EDF file at @file_name.
        @file_type is one of
            edflib.FILETYPE_EDF
            edflib.FILETYPE_EDFPLUS
            edflib.FILETYPE_BDF
            edflib.FILETYPE_BDFPLUS
        @n_channels is the number of channels without the annotation channel
        (only FILETYPE_EDFPLUS or FILETYPE_BDFPLUS)

        @channel_info should be a
        list of dicts, one for each channel in the data. Each dict needs
        these values:

            'label' : channel label (string, <= 16 characters, must be unique)
            'dimension' : physical dimension (e.g., mV) (string, <= 8 characters)
            'sample_rate' : sample frequency in hertz (int)
            'physical_max' : maximum physical value (float)
            'physical_min' : minimum physical value (float)
            'digital_max' : maximum digital value (int, -2**15 <= x < 2**15)
            'digital_min' : minimum digital value (int, -2**15 <= x < 2**15)
        '''
        self.path = file_name
        self.file_type = file_type
        self.patient_name = ''
        self.patient_code = ''
        self.technician = ''
        self.equipment = ''
        self.recording_additional = ''
        self.patient_additional = ''
        self.admincode = ''
        self.gender = 0
        self.recording_start_time = datetime.now()
        self.birthdate = date(1900,1,1)
        self.duration = 1
        self.n_channels = n_channels
        self.channels = []
        self.sample_buffer = []
        for i in np.arange(self.n_channels):
            if (file_type==FILETYPE_EDFPLUS or file_type==FILETYPE_BDFPLUS):
                self.channels.append({'label':'test_label', 'dimension':'mV', 'sample_rate':100,
                             'physical_max':1.0,'physical_min':-1.0,
                             'digital_max':8388607,'digital_min':-8388608,
                             'prefilter':'pre1','transducer':'trans1'})
            else:
                self.channels.append({'label':'test_label', 'dimension':'mV', 'sample_rate':100,
                             'physical_max':1.0,'physical_min':-1.0,
                             'digital_max':32767,'digital_min':-32768,
                             'prefilter':'pre1','transducer':'trans1'})                
            
                self.sample_buffer.append([])
        self.handle = open_file_writeonly(file_name, file_type, self.n_channels)

    def update_header(self):
        """
        Updates header to edffile struct
        """
        set_technician(self.handle, self.technician)
        set_recording_additional(self.handle, self.recording_additional)
        set_patientname(self.handle, self.patient_name)
        set_patient_additional(self.handle, self.patient_additional)
        set_equipment(self.handle, self.equipment)
        set_admincode(self.handle, self.admincode)
        set_gender(self.handle, self.gender)
        set_datarecord_duration(self.handle, self.duration)
        set_startdatetime(self.handle, self.recording_start_time.year, self.recording_start_time.month, 
                          self.recording_start_time.day, self.recording_start_time.hour, 
                          self.recording_start_time.minute, self.recording_start_time.second)
        set_birthdate(self.handle, self.birthdate.year, self.birthdate.month, self.birthdate.day)
        for i in np.arange(self.n_channels):
            set_samplefrequency(self.handle,i,self.channels[i]['sample_rate'])
            set_physical_maximum(self.handle,i,self.channels[i]['physical_max'])
            set_physical_minimum(self.handle,i,self.channels[i]['physical_min'])
            set_digital_maximum(self.handle,i,self.channels[i]['digital_max'])
            set_digital_minimum(self.handle,i,self.channels[i]['digital_min'])
            set_label(self.handle,i,self.channels[i]['label'])
            set_physical_dimension(self.handle,i,self.channels[i]['dimension'])
            set_transducer(self.handle,i,self.channels[i]['transducer'])
            set_prefilter(self.handle,i,self.channels[i]['prefilter'])

    def setChannelInfo(self,edfsignal,channel_info):
        """
        Sets the parameter for signal edfsignal.
        
        @channel_info should be a dict with
        these values:

            'label' : channel label (string, <= 16 characters, must be unique)
            'dimension' : physical dimension (e.g., mV) (string, <= 8 characters)
            'sample_rate' : sample frequency in hertz (int)
            'physical_max' : maximum physical value (float)
            'physical_min' : minimum physical value (float)
            'digital_max' : maximum digital value (int, -2**15 <= x < 2**15)
            'digital_min' : minimum digital value (int, -2**15 <= x < 2**15)
        """
        if (edfsignal < 0 or edfsignal > self.n_channels):
            raise ChannelDoesNotExist(edfsignal)        
        self.channels[edfsignal] = channel_info
        self.update_header()

    def setTechnician(self,technician):
        """
        Sets the technicians name.
        
        This function is optional and can be called only after opening a file in writemode and before the first sample write action.
        """
        self.technician = technician
        self.update_header()

    def setRecordingAdditional(self,recording_additional):
        """
        Sets the additional recordinginfo
        
        This function is optional and can be called only after opening a file in writemode and before the first sample write action.
        """
        self.recording_additional = recording_additional
        self.update_header()

    def setPatientName(self,patient_name):
        """
        Sets the patientname.
        
        This function is optional and can be called only after opening a file in writemode and before the first sample write action.
        """
        self.patient_name = patient_name
        self.update_header()

    def setPatientAdditional(self,patient_additional):
        """
        Sets the additional patientinfo.
        
        This function is optional and can be called only after opening a file in writemode and before the first sample write action.
        """
        self.technician = patient_additional
        self.update_header()

    def setEquipment(self,equipment):
        """
        Sets the name of the equipment used during the aquisition.
        
        This function is optional and can be called only after opening a file in writemode and before the first sample write action.
        """
        self.equipment = equipment
        self.update_header()

    def setAdmincode(self,admincode):
        """
        Sets the admincode.
        
        This function is optional and can be called only after opening a file in writemode and before the first sample write action.
        """
        self.admincode = admincode
        self.update_header()

    def setGender(self,gender):
        """
        Sets the gender. 1 is male, 0 is female
        
        This function is optional and can be called only after opening a file in writemode and before the first sample write action.
        """
        self.gender = gender
        self.update_header()

    def setDatarecordDuration(self,duration):
        """
        Sets the datarecord duration. The default value is 1 second.
        This function is optional, normally you don't need to change 
        the default value. The datarecord duration must be in the range 0.05 to 20.0 seconds. 
        Returns 0 on success, otherwise -1.
        
        This function is NOT REQUIRED but can be called after opening a file in writemode and 
        before the first sample write action. This function can be used when you want 
        to use a samplerate which is not an integer. For example, if you want to use 
        a samplerate of 0.5 Hz, set the samplefrequency to 5 Hz and 
        the datarecord duration to 10 seconds. Do not use this function, 
        except when absolutely necessary!
        """
        self.duration = duration
        self.update_header()    

    def setStartdatetime(self,recording_start_time):
        """
        Sets the technicians name.
        """
        self.recording_start_time = recording_start_time
        self.update_header()

    def setBirthdate(self,birthdate):
        """
        Sets the birthdate.
        
        This function is optional and can be called only after opening a file in writemode and before the first sample write action.
        """
        self.birthdate = birthdate
        self.update_header()

    def setSamplefrequency(self,edfsignal,samplefrequency):
        """
        Sets the samplefrequency of signal edfsignal.
        
        This function is required for every signal and can be called only after opening a file in writemode and before the first sample write action.
        """
        if (edfsignal < 0 or edfsignal > self.n_channels):
            raise ChannelDoesNotExist(edfsignal)
        self.channels[edfsignal]['sample_rate'] = samplefrequency
        self.update_header()

    def setPhysicalMaximum(self,edfsignal,physical_maximum):
        """
        Sets the physical_maximum of signal edfsignal.
        
        This function is required for every signal and can be called only after opening a file in writemode and before the first sample write action.
        """
        if (edfsignal < 0 or edfsignal > self.n_channels):
            raise ChannelDoesNotExist(edfsignal)
        self.channels[edfsignal]['physical_max'] = physical_maximum
        self.update_header()

    def setPhysicalMinimum(self,edfsignal,physical_minimum):
        """
        Sets the samplefrequency of signal edfsignal.
        
        This function is required for every signal and can be called only after opening a file in writemode and before the first sample write action.
        """
        if (edfsignal < 0 or edfsignal > self.n_channels):
            raise ChannelDoesNotExist(edfsignal)
        self.channels[edfsignal]['physical_min'] = physical_minimum
        self.update_header()

    def setDigitalMaximum(self,edfsignal,digital_maximum):
        """
        Sets the samplefrequency of signal edfsignal.
        Usually, the value 32767 is used for EDF+ and 8388607 for BDF+.
        
        This function is optional and can be called only after opening a file in writemode and before the first sample write action.
        """
        if (edfsignal < 0 or edfsignal > self.n_channels):
            raise ChannelDoesNotExist(edfsignal)
        self.channels[edfsignal]['digital_max'] = digital_maximum
        self.update_header()

    def setDigitalMinimum(self,edfsignal,digital_minimum):
        """
        Sets the minimum digital value of signal edfsignal.
        Usually, the value -32768 is used for EDF+ and -8388608 for BDF+. Usually this will be (-(digital_maximum + 1)).
        
        This function is optional and can be called only after opening a file in writemode and before the first sample write action.
        """
        if (edfsignal < 0 or edfsignal > self.n_channels):
            raise ChannelDoesNotExist(edfsignal)
        self.channels[edfsignal]['digital_min'] = digital_minimum
        self.update_header()

    def setLabel(self,edfsignal,label):
        """
        Sets the label (name) of signal edfsignal ("FP1", "SaO2", etc.).
        
        This function is recommended for every signal and can be called only after opening a file in writemode and before the first sample write action.
        """
        if (edfsignal < 0 or edfsignal > self.n_channels):
            raise ChannelDoesNotExist(edfsignal)
        self.channels[edfsignal]['label'] = label
        self.update_header()

    def setPhysicalDimension(self,edfsignal,physical_dimension):
        """
        Sets the physical dimension of signal edfsignal ("uV", "BPM", "mA", "Degr.", etc.)
        
        This function is recommended for every signal and can be called only after opening a file in writemode and before the first sample write action.
        """
        if (edfsignal < 0 or edfsignal > self.n_channels):
            raise ChannelDoesNotExist(edfsignal)
        self.channels[edfsignal]['dimension'] = physical_dimension
        self.update_header()

    def setTransducer(self,edfsignal,transducer):
        """
        Sets the transducer of signal edfsignal
        
        This function is optional for every signal and can be called only after opening a file in writemode and before the first sample write action.
        """
        if (edfsignal < 0 or edfsignal > self.n_channels):
            raise ChannelDoesNotExist(edfsignal)
        self.channels[edfsignal]['transducer'] = transducer
        self.update_header()

    def setPrefilter(self,edfsignal,prefilter):
        """
        Sets the prefilter of signal edfsignal ("HP:0.1Hz", "LP:75Hz N:50Hz", etc.)
        
        This function is optional for every signal and can be called only after opening a file in writemode and before the first sample write action.
        """
        if (edfsignal < 0 or edfsignal > self.n_channels):
            raise ChannelDoesNotExist(edfsignal)
        self.channels[edfsignal]['prefilter'] = prefilter
        self.update_header()

    def writePhyisicalSamples(self, data):
        """
        Writes n physical samples (uV, mA, Ohm) belonging to one signal where n 
        is the samplefrequency of the signal.
        
        @data_vec belonging to one signal. The size must be the samplefrequency of the signal.
        
        
        Writes n physical samples (uV, mA, Ohm) from data_vec belonging to one signal where n 
        is the samplefrequency of the signal. The physical samples will be converted to digital 
        samples using the values of physical maximum, physical minimum, digital maximum and digital 
        minimum. The number of samples written is equal to the samplefrequency of the signal. 
        Call this function for every signal in the file. The order is important! When there are 4 
        signals in the file, the order of calling this function must be: signal 0, signal 1, signal 2, 
        signal 3, signal 0, signal 1, signal 2, etc.
        
        All parameters must be already written into the bdf/edf-file.
        """
        write_physical_samples(self.handle, data)

    def close(self):
        """
        Closes the file.        
        """
        close_file(self.handle)
