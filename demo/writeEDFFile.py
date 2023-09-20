# Copyright (c) 2016 Holger Nahrstaedt

import os
import numpy as np
from scipy import signal
import pyedflib

# signal label/waveform  amplitude    f       sf
# ---------------------------------------------------
#    1    squarewave        100 uV    0.1Hz   200 Hz
#    2    ramp              100 uV    1 Hz    200 Hz
#    3    pulse 1           100 uV    1 Hz    200 Hz
#    4    pulse 2           100 uV    1 Hz    256 Hz
#    5    pulse 3           100 uV    1 Hz    217 Hz
#    6    noise             100 uV    - Hz    200 Hz
#    7    sine 1 Hz         100 uV    1 Hz    200 Hz
#    8    sine 8 Hz         100 uV    8 Hz    200 Hz
#    9    sine 8.1777 Hz    100 uV    8.25 Hz 200 Hz
#    10    sine 8.5 Hz       100 uV    8.5Hz   200 Hz
#    11    sine 15 Hz        100 uV   15 Hz    200 Hz
#    12    sine 17 Hz        100 uV   17 Hz    200 Hz
#    13    sine 50 Hz        100 uV   50 Hz    200 Hz


if __name__ == '__main__':
    test_data_file = os.path.join('.', 'test_generator2.edf')
    file_duration = 600
    f = pyedflib.EdfWriter(test_data_file, 13,
                           file_type=pyedflib.FILETYPE_EDFPLUS)
    channel_info = []
    data_list = []

    ch_dict = {'label': 'squarewave', 'dimension': 'uV', 'sample_frequency': 200, 'physical_max': 100, 'physical_min': -100, 'digital_max': 32767, 'digital_min': -32768, 'transducer': '', 'prefilter':''}
    channel_info.append(ch_dict)
    time = np.linspace(0, file_duration, file_duration*200)
    xtemp = np.sin(2*np.pi*0.1*time)
    x1 = xtemp.copy()
    x1[np.where(xtemp > 0)[0]] = 100
    x1[np.where(xtemp < 0)[0]] = -100
    data_list.append(x1)

    ch_dict = {'label': 'ramp', 'dimension': 'uV', 'sample_frequency': 200, 'physical_max': 100, 'physical_min': -100, 'digital_max': 32767, 'digital_min': -32768, 'transducer': '', 'prefilter':''}
    channel_info.append(ch_dict)
    time = np.linspace(0, file_duration, file_duration*200)
    x2 = signal.sawtooth(2 * np.pi * 1 * time)
    data_list.append(x2)

    ch_dict = {'label': 'pulse 1', 'dimension': 'uV', 'sample_frequency': 200, 'physical_max': 100, 'physical_min': -100, 'digital_max': 32767, 'digital_min': -32768, 'transducer': '', 'prefilter':''}
    channel_info.append(ch_dict)
    time = np.linspace(0, file_duration, file_duration*200)
    xtemp = np.sin(2*np.pi*0.5*time)
    x3 = np.zeros(file_duration*200)
    x3[np.where(np.all((xtemp < 0.02, xtemp > -0.02),axis=0))[0]] = 100
    data_list.append(x3)

    ch_dict = {'label': 'pulse 2', 'dimension': 'uV', 'sample_frequency': 256, 'physical_max': 100, 'physical_min': -100, 'digital_max': 32767, 'digital_min': -32768, 'transducer': '', 'prefilter':''}
    channel_info.append(ch_dict)
    time = np.linspace(0, file_duration, file_duration*256)
    xtemp = np.sin(2*np.pi*0.5*time)
    x4 = np.zeros(file_duration*256)
    x4[np.where(np.all((xtemp < 0.02, xtemp > -0.02),axis=0))[0]] = 100
    data_list.append(x4)

    ch_dict = {'label': 'pulse 3', 'dimension': 'uV', 'sample_frequency': 217, 'physical_max': 100, 'physical_min': -100, 'digital_max': 32767, 'digital_min': -32768, 'transducer': '', 'prefilter':''}
    channel_info.append(ch_dict)
    time = np.linspace(0, file_duration, file_duration*217)
    xtemp = np.sin(2*np.pi*0.5*time)
    x5 = np.zeros(file_duration*217)
    x5[np.where(np.all((xtemp < 0.02, xtemp > -0.02),axis=0))[0]] = 100
    data_list.append(x5)

    ch_dict = {'label': 'noise', 'dimension': 'uV', 'sample_frequency': 200, 'physical_max': 100, 'physical_min': -100, 'digital_max': 32767, 'digital_min': -32768, 'transducer': '', 'prefilter':''}
    channel_info.append(ch_dict)
    data_list.append(np.random.normal(size=file_duration*200))

    ch_dict = {'label': 'sine 1 Hz', 'dimension': 'uV', 'sample_frequency': 200, 'physical_max': 100, 'physical_min': -100, 'digital_max': 32767, 'digital_min': -32768, 'transducer': '', 'prefilter':''}
    channel_info.append(ch_dict)
    time = np.linspace(0, file_duration, file_duration*200)
    data_list.append(np.sin(2*np.pi*1*time))

    ch_dict = {'label': 'sine 8 Hz', 'dimension': 'uV', 'sample_frequency': 200, 'physical_max': 100, 'physical_min': -100, 'digital_max': 32767, 'digital_min': -32768, 'transducer': '', 'prefilter':''}
    channel_info.append(ch_dict)
    time = np.linspace(0, file_duration, file_duration*200)
    data_list.append(np.sin(2*np.pi*8*time))

    ch_dict = {'label': 'sine 8.1777 Hz', 'dimension': 'uV', 'sample_frequency': 200, 'physical_max': 100, 'physical_min': -100, 'digital_max': 32767, 'digital_min': -32768, 'transducer': '', 'prefilter':''}
    channel_info.append(ch_dict)
    time = np.linspace(0, file_duration, file_duration*200)
    data_list.append(np.sin(2*np.pi*8.1777*time))

    ch_dict = {'label': 'sine 8.5 Hz', 'dimension': 'uV', 'sample_frequency': 200, 'physical_max': 100, 'physical_min': -100, 'digital_max': 32767, 'digital_min': -32768, 'transducer': '', 'prefilter':''}
    channel_info.append(ch_dict)
    time = np.linspace(0, file_duration, file_duration*200)
    data_list.append(np.sin(2*np.pi*8.5*time))

    ch_dict = {'label': 'sine 15 Hz', 'dimension': 'uV', 'sample_frequency': 200, 'physical_max': 100, 'physical_min': -100, 'digital_max': 32767, 'digital_min': -32768, 'transducer': '', 'prefilter':''}
    channel_info.append(ch_dict)
    time = np.linspace(0, file_duration, file_duration*200)
    data_list.append(np.sin(2*np.pi*15*time))

    ch_dict = {'label': 'sine 17 Hz', 'dimension': 'uV', 'sample_frequency': 200, 'physical_max': 100, 'physical_min': -100, 'digital_max': 32767, 'digital_min': -32768, 'transducer': '', 'prefilter':''}
    channel_info.append(ch_dict)
    time = np.linspace(0, file_duration, file_duration*200)
    data_list.append(np.sin(2*np.pi*17*time))

    ch_dict = {'label': 'sine 50 Hz', 'dimension': 'uV', 'sample_frequency': 200, 'physical_max': 100, 'physical_min': -100, 'digital_max': 32767, 'digital_min': -32768, 'transducer': '', 'prefilter':''}
    channel_info.append(ch_dict)
    time = np.linspace(0, file_duration, file_duration*200)
    data_list.append(np.sin(2*np.pi*50*time))

    f.setSignalHeaders(channel_info)
    f.writeSamples(data_list)
    f.writeAnnotation(0, -1, "Recording starts")
    f.writeAnnotation(298, -1, "Test 1")
    f.writeAnnotation(294.99, -1, "pulse 1")
    f.writeAnnotation(295.9921875, -1, "pulse 2")
    f.writeAnnotation(296.99078341013825, -1, "pulse 3")
    f.writeAnnotation(600, -1, "Recording ends")
    f.close()
    del f
