
# -*- coding: utf-8 -*-
# Copyright (c) 2015 - 2017 Holger Nahrstaedt
# Copyright (c) 2011, 2015, Chris Lee-Messer
# Copyright (c) 2016-2017 The pyedflib Developers
#                         <https://github.com/holgern/pyedflib>
# See LICENSE for license details.
"""
Created on Tue Jan  7 12:13:47 2020

@author: skjerns
"""

import os
import numpy as np
import warnings
import pyedflib
from datetime import datetime
# from . import EdfWriter
# from . import EdfReader

def tqdm(*args, **kwargs):
    """
    These are optional dependecies that show a progress bar
    for some of the functions, e.g. loading.
    
    if not installed this is just a pass through iterator
    """
    try:
        from tqd2m import tqdm as iterator
        return iterator(*args, **kwargs)
    except:
        return list(args[0])
     
def _parse_date(string):
    # some common formats.
    formats = ['%Y-%m-%d', '%d-%m-%Y', '%d.%m.%Y', '%Y.%m.%d', '%d %b %Y',
               '%Y/%m/%d', '%d/%m/%Y']
    for f in formats:
        try:
            return datetime.strptime(string, f)
        except:
            pass
    print('dateparser is not installed. to convert strings to dates'\
          'install via `pip install dateparser`.')
    raise ValueError('birthdate must be datetime object or of format'\
                     ' `%d-%m-%Y`, eg. `24-01-2020`')
            
def make_header(technician='', recording_additional='', patientname='',
                patient_additional='', patientcode= '', equipment= '',
                admincode= '', gender= '', startdate=None, birthdate= ''):
    """
    A convenience function to create an EDF header (a dictionary) that
    can be used by pyedflib to update the main header of the EDF
    """
    if not birthdate=='' and isinstance(birthdate, str):
        birthdate = _parse_date(birthdate)
    if startdate is None: 
        now = datetime.now()
        startdate = datetime(now.year, now.month, now.day, 
                             now.hour, now.minute, now.second)
        del now
    if isinstance(birthdate, datetime): 
        birthdate = birthdate.strftime('%d %b %Y')
    local = locals()
    header = {}
    for var in local:
        if isinstance(local[var], datetime):
            header[var] = local[var]
        else:
            header[var] = str(local[var])
    return header


def make_signal_header(label, dimension='uV', sample_rate=256, 
                       physical_min=-200, physical_max=200, digital_min=-32768,
                       digital_max=32767, transducer='', prefiler=''):
    """
    A convenience function that creates a signal header for a given signal.
    This can be used to create a list of signal headers that is used by 
    pyedflib to create an edf. With this, different sampling frequencies 
    can be indicated.
    
    :param label: the name of the channel
    """
    signal_header = {'label': label, 
               'dimension': dimension, 
               'sample_rate': sample_rate, 
               'physical_min': physical_min, 
               'physical_max': physical_max, 
               'digital_min':  digital_min, 
               'digital_max':  digital_max, 
               'transducer': transducer, 
               'prefilter': prefiler}
    return signal_header


def make_signal_headers(list_of_labels, dimension='uV', sample_rate=256, 
                       physical_min=-200, physical_max=200, digital_min=-32768,
                       digital_max=32767, transducer='', prefiler=''):
    """
    A function that creates signal headers for a given list of channel labels.
    This can only be used if each channel has the same sampling frequency
    
    :param list_of_labels: A list with labels for each channel.
    :returns: A dictionary that can be used by pyedflib to update the header
    """
    signal_headers = []
    for label in list_of_labels:
        header = make_signal_header(label, dimension=dimension, sample_rate=sample_rate, 
                                    physical_min=physical_min, physical_max=physical_max,
                                    digital_min=digital_min, digital_max=digital_max,
                                    transducer=transducer, prefiler=prefiler)
        signal_headers.append(header)
    return signal_headers


def read_edf(edf_file, ch_nrs=None, ch_names=None, digital=False, verbose=True):
    """
    Reading EDF+/BDF data with pyedflib.

    Will load the edf and return the signals, the headers of the signals 
    and the header of the EDF. If all signals have the same sample frequency
    will return a numpy array, else a list with the individual signals
        
    :param edf_file: link to an edf file
    :param ch_nrs: The numbers of channels to read (optional)
    :param ch_names: The names of channels to read (optional)
    :returns: signals, signal_headers, header
    """      
    assert os.path.exists(edf_file), 'file {} does not exist'.format(edf_file)
    assert (ch_nrs is  None) or (ch_names is None), \
           'names xor numbers should be supplied'
    if ch_nrs is not None and not isinstance(ch_nrs, list): ch_nrs = [ch_nrs]
    if ch_names is not None and \
        not isinstance(ch_names, list): ch_names = [ch_names]

    with pyedflib.EdfReader(edf_file) as f:
        # see which channels we want to load
        available_chs = [ch.upper() for ch in f.getSignalLabels()]
        n_chrs = f.signals_in_file

        # find out which number corresponds to which channel
        if ch_names is not None:
            ch_nrs = []
            for ch in ch_names:
                if not ch.upper() in available_chs:
                    warnings.warn('{} is not in source file (contains {})'\
                                  .format(ch, available_chs))
                    print('will be ignored.')
                else:    
                    ch_nrs.append(available_chs.index(ch.upper()))
                    
        # if there ch_nrs is not given, load all channels      

        if ch_nrs is None: # no numbers means we load all
            ch_nrs = range(n_chrs)
        
        # convert negative numbers into positives
        ch_nrs = [n_chrs+ch if ch<0 else ch for ch in ch_nrs]
        
        # load headers, signal information and 
        header = f.getHeader()
        signal_headers = [f.getSignalHeaders()[c] for c in ch_nrs]

        signals = []
        for i,c in enumerate(tqdm(ch_nrs, desc='Reading Channels', 
                                  disable=not verbose)):
            signal = f.readSignal(c, digital=digital)
            signals.append(signal)
 
        # we can only return a np.array if all signals have the same samplefreq           
        sfreqs = [header['sample_rate'] for header in signal_headers]
        all_sfreq_same = sfreqs[1:]==sfreqs[:-1]
        if all_sfreq_same:
            dtype = np.int if digital else np.float
            signals = np.array(signals, dtype=dtype)
        elif verbose:
            warnings.warn('Not all sampling frequencies are the same ({}). '\
                          .format(sfreqs))    
    assert len(signals)==len(signal_headers), 'Something went wrong, lengths'\
                                         ' of headers is not length of signals'
    return  signals, signal_headers, header


def write_edf(edf_file, signals, signal_headers, header, digital=False):
    """
    Write signals to an edf_file. Header can be generated on the fly.
    
    :param signals: The signals as a list of arrays or a ndarray
    :param signal_headers: a list with one signal header(dict) for each signal.
                           See pyedflib.EdfWriter.setSignalHeader
    :param header: a main header (dict) for the EDF file, see 
                   pyedflib.EdfWriter.setHeader for details
    :param digital: whether signals are presented digitally 
                    or in physical values
    
    :returns: True if successful, False if failed
    """
    assert header is None or isinstance(header, dict), \
        'header must be dictioniary'
    assert isinstance(signal_headers, list), \
        'signal headers must be list'
    assert len(signal_headers)==len(signals), \
        'signals and signal_headers must be same length'
        
    n_channels = len(signals)
    
    with pyedflib.EdfWriter(edf_file, n_channels=n_channels) as f:  
        f.setSignalHeaders(signal_headers)
        f.setHeader(header)
        f.writeSamples(signals, digital=digital)
        
    return os.path.isfile(edf_file) 


def write_edf_quick(edf_file, signals, sfreq, digital=False):
    """
    wrapper for write_pyedf without creating headers.
    Use this if you don't care about headers or channel names and just
    want to dump some signals with the same sampling freq. to an edf
    
    :param edf_file: where to store the data/edf
    :param signals: The signals you want to store as numpy array
    :param sfreq: the sampling frequency of the signals
    :param digital: if the data is present digitally (int) or as mV/uV
    """
    labels = ['CH_{}'.format(i) for i in range(len(signals))]
    signal_headers = make_signal_headers(labels, sample_rate = sfreq)
    return write_edf(edf_file, signals, signal_headers, digital=digital)


def read_edf_header(edf_file):
    """
    Reads the header and signal headers of an EDF file
    
    :returns: header of the edf file (dict)
    """
    assert os.path.isfile(edf_file), 'file {} does not exist'.format(edf_file)
    with pyedflib.EdfReader(edf_file) as f:
        summary = f.getHeader()
        summary['Duration'] = f.getFileDuration
        summary['SignalHeaders'] = f.getSignalHeaders()
        summary['channels'] = f.getSignalLabels()
    del f
    return summary



def drop_channels(edf_source, edf_target=None, to_keep=None, to_drop=None):
    """
    Remove channels from an edf file using pyedflib.
    Save the file as edf_target. 
    For safety reasons, no source files can be overwritten.
    
    :param edf_source: The source edf file
    :param edf_target: Where to save the file. 
                       If None, will be edf_source+'dropped.edf'
    :param to_keep: A list of channel names or indices that will be kept.
                    Strings will always be interpreted as channel names.
                    'to_keep' will overwrite any droppings proposed by to_drop
    :param to_drop: A list of channel names/indices that should be dropped.
                    Strings will be interpreted as channel names.
    :returns: the target filename with the dropped channels
    """
    # convert to list if necessary
    if isinstance(to_keep, (int, str)): to_keep = [to_keep]
    if isinstance(to_drop, (int, str)): to_drop = [to_drop]
    
    # check all parameters are good
    assert to_keep is None or to_drop is None,'Supply only to_keep xor to_drop'
    if to_keep is not None:
        assert all([isinstance(ch, (str, int)) for ch in to_keep]),\
            'channels must be int or string'
    if to_drop is not None:
        assert all([isinstance(ch, (str, int)) for ch in to_drop]),\
            'channels must be int or string'
    assert os.path.exists(edf_source), 'source file {} does not exist'\
                                       .format(edf_source)
    assert edf_source!=edf_target, 'For safet, target must not be source file.'
        
    if edf_target is None: 
        edf_target = os.path.splitext(edf_source)[0] + '_dropped.edf'
    if os.path.exists(edf_target): 
        warnings.warn('Target file will be overwritten')
    
    ch_names = read_edf_header(edf_source)['channels']
    # convert to all lowercase for compatibility
    ch_names = [ch.lower() for ch in ch_names]
    ch_nrs = list(range(len(ch_names)))
    
    if to_keep is not None:
        for i,ch in enumerate(to_keep):
            if isinstance(ch,str):
                ch_idx = ch_names.index(ch.lower())
                to_keep[i] = ch_idx
        load_channels = to_keep.copy()
    elif to_drop is not None:
        for i,ch in enumerate(to_drop):
            if isinstance(ch,str):
                ch_idx = ch_names.index(ch.lower())
                to_drop[i] = ch_idx 
        to_drop = [len(ch_nrs)+ch if ch<0 else ch for ch in to_drop]

        [ch_nrs.remove(ch) for ch in to_drop]
        load_channels = ch_nrs.copy()
    else:
        raise ValueError
        
    signals, signal_headers, header = read_edf(edf_source, 
                                               ch_nrs=load_channels, 
                                               digital=True)
    
    write_edf(edf_target, signals, signal_headers, header, digital=True)
    return edf_target


def anonymize_edf(edf_file, new_file=None, 
                  to_remove   = ['patientname', 'birthdate'],
                  new_values  = ['xxx', '']):
    """
    Anonymizes an EDF file, that means it strips all header information
    that is patient specific, ie. birthdate and patientname as well as XXX
    
    :param edf_file: a string with a filename of an EDF/BDF
    :param new_file: where to save the anonymized edf file
    :param to_remove: a list of attributes to remove from the file
    :param new_values: a list of values that should be given instead to the edf
    :returns: True if successful, False if failed
    """
    assert len(to_remove)==len(new_values), \
           'Each to_remove must have one new_value'
    header = read_edf_header(edf_file)
    
    for new_val, attr in zip(new_values, to_remove):
        header[attr] = new_val
        
    if new_file is None:
        file, ext = os.path.splitext(edf_file)
        new_file = file + '_anonymized' + ext
    n_chs = len(header['channels'])
    signal_headers = []
    signals = []
    for ch_nr in tqdm(range(n_chs)):
        signal, signal_header, _ = read_edf(edf_file, digital=True, 
                                            ch_nrs=ch_nr, verbose=False)
        signal_headers.append(signal_header[0])
        signals.append(signal.squeeze())

    return write_edf(new_file, signals, signal_headers, header,digital=True)


def rename_channels(edf_file, mapping, new_file=None):
    """
    A convenience function to rename channels in an EDF file.
    
    :param edf_file: an string pointing to an edf file
    :param mapping:  a dictionary with channel mappings as key:value
    :param new_file: the new filename
    """
    header = read_edf_header(edf_file)
    channels = header['channels']
    if new_file is None:
        file, ext = os.path.splitext(edf_file)
        new_file = file + '_renamed' + ext

    signal_headers = []
    signals = []
    for ch_nr in tqdm(range(len(channels))):
        signal, signal_header, _ = read_edf(file, digital=True, 
                                            ch_nrs=ch_nr, verbose=False)
        ch = signal_header[0]['label']
        if ch in mapping :
            print('{} to {}'.format(ch, mapping[ch]))
            ch = mapping[ch]
            signal_header[0]['label']=ch
        else:
            print('no mapping for {}, leave as it is'.format(ch))
        signal_headers.append(signal_header[0])
        signals.append(signal.squeeze())

    write_edf(new_file, signals, signal_headers, header,digital=True)
    