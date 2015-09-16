# -*- coding: utf-8 -*-
# Copyright (c) 2015 Chris Lee-Messer <https://bitbucket.org/cleemesser/>
# Copyright (c) 2015 Holger Nahrstaedt

from __future__ import division, print_function, absolute_import

__all__ = ['EdfWriter']

import numpy as np

from ._edflib import *


class EdfWriter(object):
    def __init__(self, file_name, channel_info,
                 file_type=FILETYPE_EDFPLUS, **kwargs):
        '''Initialises an EDF file at @file_name.
        @file_type is one of
            edflib.FILETYPE_EDF
            edflib.FILETYPE_EDFPLUS
            edflib.FILETYPE_BDF
            edflib.FILETYPE_BDFPLUS

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
        self.n_channels = len(channel_info)
        self.channels = {}
        for c in channel_info:
            if c['label'] in self.channels:
                raise ChannelLabelExists(c['label'])
            self.channels[c['label']] = c
        self.sample_buffer = dict([(c['label'], []) for c in channel_info])
        self.handle = open_file_writeonly(file_name, file_type, self.n_channels)
        self._init_constants(**kwargs)
        self._init_channels(channel_info)

    def write_sample(self, channel_label, sample):
        '''Queues a digital sample for @channel_label for recording; the data won't
        actually be written until one second's worth of data has been queued.'''
        if channel_label not in self.channels:
            raise ChannelDoesNotExist(channel_label)
        self.sample_buffer[channel_label].append(sample)
        if len(self.sample_buffer[channel_label]) == self.channels[channel_label]['sample_rate']:
            self._flush_samples()

    def close(self):
        close_file(self.handle)

    def _init_constants(self, **kwargs):
        def call_if_set(fn, kw_name):
            item = kwargs.pop(kw_name, None)
            if item is not None:
                fn(self.handle, item)
        call_if_set(set_technician, 'technician')
        call_if_set(set_recording_additional, 'recording_additional')
        call_if_set(set_patientname, 'patient_name')
        call_if_set(set_patient_additional, 'patient_additional')
        call_if_set(set_equipment, 'equipment')
        call_if_set(set_admincode, 'admincode')
        call_if_set(set_gender, 'gender')
        call_if_set(set_datarecord_duration, 'duration')
        call_if_set((lambda hdl, dt: set_startdatetime(hdl, dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)), 'recording_start_time')
        call_if_set((lambda hdl, dt: set_birthdate(hdl, dt.year, dt.month, dt.day)), 'patient_birthdate')
        if len(kwargs) > 0:
            raise Exception('Unhandled argument(s) given: %r' % kwargs.keys())

    def _init_channels(self, channels):
        hdl = self.handle

        def call_per_channel(fn, name, optional=False):
            for i,c in enumerate(channels):
                if optional and not (name in c):
                    continue
                fn(hdl, i, c.pop(name))

        call_per_channel(set_samplefrequency, 'sample_rate')
        call_per_channel(set_physical_maximum, 'physical_max')
        call_per_channel(set_digital_maximum, 'digital_max')
        call_per_channel(set_digital_minimum, 'digital_min')
        call_per_channel(set_physical_minimum, 'physical_min')
        call_per_channel(set_label, 'label')
        call_per_channel(set_physical_dimension, 'dimension')
        call_per_channel(set_transducer, 'transducer', optional=True)
        call_per_channel(set_prefilter, 'prefilter', optional=True)

    def _flush_samples(self):
        for c in self.channels:
            buf = np.array(self.sample_buffer[c], dtype='int16')
            write_digital_samples(self.handle, buf)
            self.sample_buffer[c] = []