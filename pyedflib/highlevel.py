# Copyright (c) 2019 - 2023 Simon Kern
# Copyright (c) 2015 - 2023 Holger Nahrstaedt
# Copyright (c) 2011, 2015, Chris Lee-Messer
# Copyright (c) 2016-2017 The pyedflib Developers
#                         <https://github.com/holgern/pyedflib>
# See LICENSE for license details.
"""
Created on Tue Jan  7 12:13:47 2020

This file contains high-level functions to work with pyedflib.

Includes
    - Reading and writing EDFs
    - Anonymizing EDFs
    - Comparing EDFs
    - Renaming Channels from EDF files
    - Dropping Channels from EDF files
    - Cropping EDFs

@author: skjerns
"""

import os
import numpy as np
import warnings
import pyedflib
from copy import deepcopy
from datetime import datetime, timedelta
# from . import EdfWriter
# from . import EdfReader


def _get_sample_frequency(signal_header):
    # Temporary conditional assignment while we deprecate 'sample_rate' as a channel attribute
    # in favor of 'sample_frequency', supporting the use of either to give
    # users time to switch to the new interface.
    return (signal_header['sample_rate']
            if signal_header.get('sample_frequency') is None
            else signal_header['sample_frequency'])


def tqdm(iterable, *args, **kwargs):
    """
    These is an optional dependency that shows a progress bar for some
    of the functions, e.g. loading.

    install this dependency with `pip install tqdm`

    if not installed this is just a pass through iterator.
    """
    try:
        from tqdm import tqdm as iterator
        return iterator(iterable, *args, **kwargs)
    except:
        return iterable


def _parse_date(string):
    """
    A simple dateparser that detects common  date formats

    Parameters
    ----------
    string : str
        a date string in format as denoted below.

    Returns
    -------
    datetime.datetime
        datetime object of a time.

    """
    # some common formats.
    formats = ['%Y-%m-%d', '%d-%m-%Y', '%d.%m.%Y', '%Y.%m.%d', '%d %b %Y',
               '%Y/%m/%d', '%d/%m/%Y']
    for f in formats:
        try:
            return datetime.strptime(string, f)
        except:
            pass
    try:
        import dateparser
        return dateparser.parse(string)
    except:
        print('dateparser is not installed. to convert strings to dates'\
              'install via `pip install dateparser`.')
        raise ValueError('birthdate must be datetime object or of format'\
                         ' `%d-%m-%Y`, eg. `24-01-2020`')

def dig2phys(signal, dmin, dmax, pmin, pmax):
    """
    converts digital edf values to physical values

    Parameters
    ----------
    signal : np.ndarray or int
        A numpy array with int values (digital values) or an int.
    dmin : int
        digital minimum value of the edf file (eg -2048).
    dmax : int
        digital maximum value of the edf file (eg 2048).
    pmin : float
        physical maximum value of the edf file (eg -200.0).
    pmax : float
        physical maximum value of the edf file (eg 200.0).

    Returns
    -------
    physical : np.ndarray or float
        converted physical values

    """
    m = (pmax-pmin) / (dmax-dmin)
    b = pmax / m - dmax
    physical = m * (signal + b)
    return physical


def phys2dig(signal, dmin, dmax, pmin, pmax):
    """
    converts physical values to digital values

    Parameters
    ----------
    signal : np.ndarray or int
        A numpy array with int values (digital values) or an int.
    dmin : int
        digital minimum value of the edf file (eg -2048).
    dmax : int
        digital maximum value of the edf file (eg 2048).
    pmin : float
        physical maximum value of the edf file (eg -200.0).
    pmax : float
        physical maximum value of the edf file (eg 200.0).

    Returns
    -------
    digital : np.ndarray or int
        converted digital values

    """
    m = (pmax-pmin) / (dmax-dmin)
    b = pmax / m - dmax
    digital = signal/m - b
    return digital



def make_header(technician='', recording_additional='', patientname='',
                patient_additional='', patientcode= '', equipment= '',
                admincode= '', sex= '', startdate=None, birthdate= '',
                gender=None):
    """
    A convenience function to create an EDF header (a dictionary) that
    can be used by pyedflib to update the main header of the EDF

    Parameters
    ----------
    technician : str, optional
        name of the technician. The default is ''.
    recording_additional : str, optional
        comments etc. The default is ''.
    patientname : str, optional
        the name of the patient. The default is ''.
    patient_additional : TYPE, optional
        more info about the patient. The default is ''.
    patientcode : str, optional
        alphanumeric code. The default is ''.
    equipment : str, optional
        which system was used. The default is ''.
    admincode : str, optional
        code of the admin. The default is ''.
    sex : str, optional
        sex of patient. The default is ''.
    startdate : datetime.datetime, optional
        startdate of recording. The default is None.
    birthdate : str/datetime.datetime, optional
        date of birth of the patient. The default is ''.

    Returns
    -------
    header : dict
        a dictionary with the values given filled in.

    """

    if not birthdate=='' and isinstance(birthdate, str):
        birthdate = _parse_date(birthdate)
    if startdate is None:
        now = datetime.now()
        startdate = datetime(now.year, now.month, now.day,
                             now.hour, now.minute, now.second)
        del now
    if isinstance(birthdate, datetime):
        birthdate = birthdate.strftime('%d %b %Y').lower()

    # backwards compatibility
    if gender is not None:
        if sex == '':
            sex = gender
            warnings.warn("Parameter 'gender' is deprecated, use 'sex' instead.", DeprecationWarning, stacklevel=2)
        elif sex != gender:
            raise ValueError("Defined both parameters 'sex' and 'gender', with different values: {sex} != {gender}")
    gender = sex

    local = locals()

    header = {}
    for var in local:
        if isinstance(local[var], datetime):
            header[var] = local[var]
        else:
            header[var] = str(local[var])
    return header


def make_signal_header(label, dimension='uV', sample_rate=256, sample_frequency=None,
                       physical_min=-200, physical_max=200, digital_min=-32768,
                       digital_max=32767, transducer='', prefiler=''):
    """
    A convenience function that creates a signal header for a given signal.
    This can be used to create a list of signal headers that is used by
    pyedflib to create an edf. With this, different sampling frequencies
    can be indicated.

    Parameters
    ----------
    label : str
        the name of the channel.
    dimension : str, optional
        dimension, eg mV. The default is 'uV'.
    sample_rate : int, optional
        sampling frequency. The default is 256. Deprecated: use 'sample_frequency' instead.
    sample_frequency : int, optional
        sampling frequency. The default is 256.
    physical_min : float, optional
        minimum value in dimension. The default is -200.
    physical_max : float, optional
        maximum value in dimension. The default is 200.
    digital_min : int, optional
        digital minimum of the ADC. The default is -32768.
    digital_max : int, optional
        digital maximum of the ADC. The default is 32767.
    transducer : str, optional
        electrode type that was used. The default is ''.
    prefiler : str, optional
        filtering and sampling method. The default is ''.

    Returns
    -------
    signal_header : dict
        a signal header that can be used to save a channel to an EDF.

    """

    signal_header = {'label': label,
               'dimension': dimension,
               'sample_rate': sample_rate,
               'sample_frequency': sample_frequency,
               'physical_min': physical_min,
               'physical_max': physical_max,
               'digital_min':  digital_min,
               'digital_max':  digital_max,
               'transducer': transducer,
               'prefilter': prefiler}
    return signal_header


def make_signal_headers(list_of_labels, dimension='uV', sample_rate=256,
                       sample_frequency=None, physical_min=-200.0, physical_max=200.0,
                       digital_min=-32768, digital_max=32767,
                       transducer='', prefiler=''):
    """
    A function that creates signal headers for a given list of channel labels.
    This can only be used if each channel has the same sampling frequency

    Parameters
    ----------
    list_of_labels : list of str
        A list with labels for each channel.
    dimension : str, optional
        dimension, eg mV. The default is 'uV'.
    sample_rate : int, optional
        sampling frequency. The default is 256.  Deprecated: use 'sample_frequency' instead.
    sample_frequency : int, optional
        sampling frequency. The default is 256.
    physical_min : float, optional
        minimum value in dimension. The default is -200.
    physical_max : float, optional
        maximum value in dimension. The default is 200.
    digital_min : int, optional
        digital minimum of the ADC. The default is -32768.
    digital_max : int, optional
        digital maximum of the ADC. The default is 32767.
    transducer : str, optional
        electrode type that was used. The default is ''.
    prefiler : str, optional
        filtering and sampling method. The default is ''.

    Returns
    -------
    signal_headers : list of dict
        returns n signal headers as a list to save several signal headers.

    """
    signal_headers = []
    for label in list_of_labels:
        header = make_signal_header(label, dimension=dimension, sample_rate=sample_rate,
                                    sample_frequency=sample_frequency,
                                    physical_min=physical_min, physical_max=physical_max,
                                    digital_min=digital_min, digital_max=digital_max,
                                    transducer=transducer, prefiler=prefiler)
        signal_headers.append(header)
    return signal_headers


def read_edf(edf_file, ch_nrs=None, ch_names=None, digital=False, verbose=False):
    """
    Convenience function for reading EDF+/BDF data with pyedflib.

    Will load the edf and return the signals, the headers of the signals
    and the header of the EDF. If all signals have the same sample frequency
    will return a numpy array, else a list with the individual signals


    Parameters
    ----------
    edf_file : str
        link to an edf file.
    ch_nrs : list of int, optional
        The indices of the channels to read. The default is None.
    ch_names : list of str, optional
        The names of channels to read. The default is None.
    digital : bool, optional
        will return the signals as digital values (ADC). The default is False.
    verbose : bool, optional
        Print progress bar while loading or not. The default is False.

    Returns
    -------
    signals : np.ndarray or list
        the signals of the chosen channels contained in the EDF.
    signal_headers : list
        one signal header for each channel in the EDF.
    header : dict
        the main header of the EDF file containing meta information.

    """
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

        # add annotations to header
        annotations = f.readAnnotations()
        annotations = [[s, d, a] for s,d,a in zip(*annotations)]
        header['annotations'] = annotations


        signals = []
        for i,c in enumerate(tqdm(ch_nrs, desc='Reading Channels',
                                  disable=not verbose)):
            signal = f.readSignal(c, digital=digital)
            signals.append(signal)

        # we can only return a np.array if all signals have the same samplefreq
        sfreqs = [_get_sample_frequency(shead) for shead in signal_headers]
        all_sfreq_same = sfreqs[1:]==sfreqs[:-1]
        if all_sfreq_same:
            dtype = np.int32 if digital else float
            signals = np.array(signals, dtype=dtype)

    assert len(signals)==len(signal_headers), 'Something went wrong, lengths'\
                                         ' of headers is not length of signals'
    del f
    return  signals, signal_headers, header


def write_edf(edf_file, signals, signal_headers, header=None, digital=False,
              file_type=-1):
    """
    Write signals to an edf_file. Header can be generated on the fly with
    generic values. EDF+/BDF+ is selected based on the filename extension,
    but can be overwritten by setting file_type to pyedflib.FILETYPE_XXX

    Parameters
    ----------
    edf_file : np.ndarray or list
        where to save the EDF file
    signals : list
        The signals as a list of arrays or a ndarray.

    signal_headers : list of dict
        a list with one signal header(dict) for each signal.
        See pyedflib.EdfWriter.setSignalHeader..
    header : dict
        a main header (dict) for the EDF file, see
        pyedflib.EdfWriter.setHeader for details.
        If no header present, will create an empty header
    digital : bool, optional
        whether the signals are in digital format (ADC). The default is False.
    file_type: int, optional
        choose file_type for saving.
        EDF = 0, EDF+ = 1, BDF = 2, BDF+ = 3, automatic from extension = -1

    Returns
    -------
    bool
         True if successful, False if failed.
    """
    assert header is None or isinstance(header, dict), \
        'header must be dictioniary or None'
    assert isinstance(signal_headers, list), \
        'signal headers must be list'
    assert len(signal_headers)==len(signals), \
        'signals and signal_headers must be same length'
    assert file_type in [-1, 0, 1, 2, 3], \
        'file_type must be in range -1, 3'

    # copy objects to prevent accidental changes to mutable objects
    header = deepcopy(header)
    signal_headers = deepcopy(signal_headers)

    if file_type==-1:
        ext = os.path.splitext(edf_file)[-1]
        if ext.lower() == '.edf':
            file_type = pyedflib.FILETYPE_EDFPLUS
        elif ext.lower() == '.bdf':
            file_type = pyedflib.FILETYPE_BDFPLUS
        else:
            raise ValueError(f'Unknown extension {ext}')

    n_channels = len(signals)

    # if there is no header, we create one with dummy values
    if header is None:
        header = {}
    default_header = make_header()
    default_header.update(header)
    header = default_header

    # check dmin, dmax and pmin, pmax dont exceed signal min/max
    for sig, shead in zip(signals, signal_headers):
        dmin, dmax = shead['digital_min'], shead['digital_max']
        pmin, pmax = shead['physical_min'], shead['physical_max']
        label = shead['label']
        if digital: # exception as it will lead to clipping
            assert dmin<=sig.min(), \
            'digital_min is {}, but signal_min is {}' \
            'for channel {}'.format(dmin, sig.min(), label)
            assert dmax>=sig.max(), \
            'digital_min is {}, but signal_min is {}' \
            'for channel {}'.format(dmax, sig.max(), label)
            assert pmin != pmax, \
            f'physical_min {pmin} should be different from physical_max {pmax}'
        else: # only warning, as this will not lead to clipping
            assert pmin<=sig.min(), \
            'phys_min is {}, but signal_min is {} ' \
            'for channel {}'.format(pmin, sig.min(), label)
            assert pmax>=sig.max(), \
            'phys_max is {}, but signal_max is {} ' \
            'for channel {}'.format(pmax, sig.max(), label)

    # get annotations, in format [[timepoint, duration, description], [...]]
    annotations = header.get('annotations', [])

    with pyedflib.EdfWriter(edf_file, n_channels=n_channels, file_type=file_type) as f:
        f.setSignalHeaders(signal_headers)
        f.setHeader(header)
        f.writeSamples(signals, digital=digital)
        for annotation in annotations:
            f.writeAnnotation(*annotation)
    del f

    return os.path.isfile(edf_file)


def write_edf_quick(edf_file, signals, sfreq, digital=False):
    """
    wrapper for write_pyedf without creating headers.
    Use this if you don't care about headers or channel names and just
    want to dump some signals with the same sampling freq. to an edf

    Parameters
    ----------
    edf_file : str
        where to store the data/edf.
    signals : np.ndarray
        The signals you want to store as numpy array.
    sfreq : int
        the sampling frequency of the signals.
    digital : bool, optional
        if the data is present digitally (int) or as mV/uV.The default is False.

    Returns
    -------
    bool
        True if successful, else False or raise Error.

    """
    signals = np.atleast_2d(signals)
    header = make_header(technician='pyedflib-quickwrite')
    labels = [f'CH_{i}' for i in range(len(signals))]
    pmin, pmax = signals.min(), signals.max()
    signal_headers = make_signal_headers(labels, sample_frequency=sfreq,
                                         physical_min=pmin, physical_max=pmax)
    return write_edf(edf_file, signals, signal_headers, header, digital=digital)


def read_edf_header(edf_file, read_annotations=True):
    """
    Reads the header and signal headers of an EDF file and it's annotations

    Parameters
    ----------
    edf_file : str
        EDF/BDF file to read.

    Returns
    -------
    summary : dict
        header of the edf file as dictionary.

    """
    assert os.path.isfile(edf_file), f'file {edf_file} does not exist'
    with pyedflib.EdfReader(edf_file) as f:

        summary = f.getHeader()
        summary['Duration'] = f.getFileDuration()
        summary['SignalHeaders'] = f.getSignalHeaders()
        summary['channels'] = f.getSignalLabels()
        if read_annotations:
            annotations = f.read_annotation()
            annotations = [[float(t)/10000000, d if d else -1, x.decode()] for t,d,x in annotations]
            summary['annotations'] = annotations
    del f
    return summary


def compare_edf(edf_file1, edf_file2, verbose=False):
    """
    Loads two edf files and checks whether the values contained in
    them are the same. Does not check the header or annotations data.

    Mainly to verify that other options (eg anonymization) produce the
    same EDF file.

    Parameters
    ----------
    edf_file1 : str
        edf file 1 to compare.
    edf_file2 : str
        edf file 2 to compare.
    verbose : bool, optional
        print progress or not. The default is False.

    Returns
    -------
    bool
        True if signals are equal, else raises error.
    """
    signals1, shead1, _ =  read_edf(edf_file1, digital=True, verbose=verbose)
    signals2, shead2, _ =  read_edf(edf_file2, digital=True, verbose=verbose)

    for i, sigs in enumerate(zip(signals1, signals2)):
        s1, s2 = sigs
        if np.array_equal(s1, s2): continue # early stopping
        s1 = np.abs(s1)
        s2 = np.abs(s2)
        if np.array_equal(s1, s2): continue # early stopping
        close =  np.mean(np.isclose(s1, s2))
        assert close>0.99, 'Error, digital values of {}'\
              ' and {} for ch {}: {} are not the same: {:.3f}'.format(
                edf_file1, edf_file2, shead1[i]['label'],
                shead2[i]['label'], close)

    dmin1, dmax1 = shead1[i]['digital_min'], shead1[i]['digital_max']
    pmin1, pmax1 = shead1[i]['physical_min'], shead1[i]['physical_max']
    dmin2, dmax2 = shead2[i]['digital_min'], shead2[i]['digital_max']
    pmin2, pmax2 = shead2[i]['physical_min'], shead2[i]['physical_max']

    for i, sigs in enumerate(zip(signals1, signals2)):
        s1, s2 = sigs

        # convert to physical values, no need to load all data again
        s1 = dig2phys(s1, dmin1, dmax1, pmin1, pmax1)
        s2 = dig2phys(s2, dmin2, dmax2, pmin2, pmax2)

        # compare absolutes in case of inverted signals
        if np.array_equal(s1, s2): continue # early stopping
        s1 = np.abs(s1)
        s2 = np.abs(s2)
        if np.array_equal(s1, s2): continue # early stopping
        min_dist = np.abs(dig2phys(1, dmin1, dmax1, pmin1, pmax1))
        close =  np.mean(np.isclose(s1, s2, atol=min_dist))
        assert close>0.99, 'Error, physical values of {}'\
            ' and {} for ch {}: {} are not the same: {:.3f}'.format(
                edf_file1, edf_file2, shead1[i]['label'],
                shead2[i]['label'], close)
    return True


def drop_channels(edf_source, edf_target=None, to_keep=None, to_drop=None,
                  verbose=False):
    """
    Remove channels from an edf file. Save the file.
    For safety reasons, no source files can be overwritten.

    Parameters
    ----------
    edf_source : str
        The source edf file from which to drop channels.
    edf_target : str, optional
        Where to save the file.If None, will be edf_source+'dropped.edf'.
        The default is None.
    to_keep : list, optional
         A list of channel names or indices that will be kept.
         Strings will always be interpreted as channel names.
         'to_keep' will overwrite any droppings proposed by to_drop.
         The default is None.
    to_drop : list, optional
        A list of channel names/indices that should be dropped.
        Strings will be interpreted as channel names. The default is None.
    verbose : bool, optional
        print progress or not. The default is False.

    Returns
    -------
    edf_target : str
         the target filename with the dropped channels.

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
    assert os.path.exists(edf_source), \
            f'source file {edf_source} does not exist'
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
        load_channels = list(to_keep) # copy list compatible with py2.7
    elif to_drop is not None:
        for i,ch in enumerate(to_drop):
            if isinstance(ch,str):
                ch_idx = ch_names.index(ch.lower())
                to_drop[i] = ch_idx
        to_drop = [len(ch_nrs)+ch if ch<0 else ch for ch in to_drop]

        [ch_nrs.remove(ch) for ch in to_drop]
        load_channels = list(ch_nrs)
    else:
        raise ValueError

    signals, signal_headers, header = read_edf(edf_source,
                                               ch_nrs=load_channels,
                                               digital=True, verbose=verbose)

    write_edf(edf_target, signals, signal_headers, header, digital=True)
    return edf_target


def anonymize_edf(edf_file, new_file=None,
                  to_remove=['patientname', 'birthdate'],
                  new_values=['xxx', ''], verify=False, verbose=False):
    """Anonymize an EDF file by replacing values of header fields.

    This function can be used to overwrite all header information that is
    patient specific, for example birthdate and patientname. All header fields
    can be overwritten this way (i.e., all header.keys() given
    _, _, header = read_edf(edf_file, digital=True)).

    Parameters
    ----------
    edf_file : str
         Filename of an EDF/BDF.
    new_file : str | None
         The filename of the anonymized file. If None, the input filename
         appended with '_anonymized' is used. Defaults to None.
    to_remove : list of str
        List of attributes to overwrite in the `edf_file`. Defaults to
        ['patientname', 'birthdate'].
    new_values : list of str
        List of values used for overwriting the attributes specified in
        `to_remove`. Each item in `to_remove` must have a corresponding item
        in `new_values`. Defaults to ['xxx', ''].
    verify : bool
        Compare `edf_file` and `new_file` for equality (i.e., double check that
        values are same). Defaults to False
    verbose : bool, optional
        print progress or not. The default is False.

    Returns
    -------
    bool
        True if successful, or if `verify` is False. Raises an error otherwise.

    """
    if not len(to_remove) == len(new_values):
        raise AssertionError('Each `to_remove` must have one `new_value`')

    if new_file is None:
        file, ext = os.path.splitext(edf_file)
        new_file = file + '_anonymized' + ext

    signals, signal_headers, header = read_edf(edf_file, digital=True,
                                               verbose=verbose)

    for new_val, attr in zip(new_values, to_remove):
        header[attr] = new_val

    write_edf(new_file, signals, signal_headers, header, digital=True)
    if verify:
        compare_edf(edf_file, new_file, verbose=verbose)
    return True


def crop_edf(
    edf_file,
    *,
    new_file=None,
    start=None,
    stop=None,
    start_format="datetime",
    stop_format="datetime",
    verbose=True,
):
    """Crop an EDF file to desired start/stop times.

    The new start/end times can be either specified as a datetime.datetime or
    as seconds from the beginning of the recording.
    For example, using `crop_edf(..., start=10, start_format="seconds") will
    remove the first 10-seconds of the recording.

    Parameters
    ----------
    edf_file : str
        The path to the EDF file.
    new_file : str | None
         The path to the new cropped file. If None (default), the input
         filename appended with '_cropped' is used.
    start : datetime.datetime | int | float | None
        The new start. Can be None to keep the original start time of
        the recording.
    stop : datetime.datetime | int | float | None
        The new stop. Can be None to keep the original end time of the
        recording.
    start_format : str
        The format of ``start``: "datetime" (default) or "seconds".
    stop_format : str
        The format of ``stop``: "datetime" (default) or "seconds".
    verbose : bool
        If True (default), print some details about the original and cropped
        file.
    """
    # Check input
    assert start_format in ["datetime", "seconds"]
    assert stop_format in ["datetime", "seconds"]
    if start_format == "datetime":
        assert isinstance(start, (datetime, type(None)))
    else:
        assert isinstance(start, (int, float, type(None)))
    if stop_format == "datetime":
        assert isinstance(stop, (datetime, type(None)))
    else:
        assert isinstance(start, (int, float, type(None)))

    # Open the original EDF file
    edf = pyedflib.EdfReader(edf_file)
    signals_headers = edf.getSignalHeaders()
    header = edf.getHeader()

    # Define new start time
    current_start = edf.getStartdatetime()
    if start is None:
        start = current_start
    else:
        if start_format == "seconds":
            start = current_start + timedelta(seconds=start)
        else:
            pass
    assert current_start <= start, 'start must not be before current start of recording'
    start_diff_from_start = (start - current_start).total_seconds()

    # Define new stop time
    current_stop = current_start + timedelta(seconds=edf.getFileDuration())
    current_duration = current_stop - current_start
    if stop is None:
        stop = current_stop
    else:
        if stop_format == "seconds":
            stop = current_start + timedelta(seconds=stop)
        else:
            pass
    assert stop <= current_stop, 'new stop value must not be after current end of recording'
    
    assert start < current_stop, 'new start value must not be after current end of recording'
    assert stop > current_start, 'new stop value must not be before current start of recording'
    stop_diff_from_start = (stop - current_start).total_seconds()

    # Crop each signal
    signals = []
    for i in range(len(edf.getSignalHeaders())):
        sf = edf.getSampleFrequency(i)
        # Convert from seconds to samples
        start_idx = int(np.round(start_diff_from_start * sf))
        stop_idx = int(np.round(stop_diff_from_start * sf))
        # We use digital=True in reading and writing to avoid precision loss
        signals.append(
            edf.readSignal(i, start=start_idx, n=stop_idx - start_idx, digital=True)
        )
    edf.close()

    # Update header startdate and save file
    header["startdate"] = start
    if new_file is None:
        file, ext = os.path.splitext(edf_file)
        new_file = file + "_cropped" + ext
    write_edf(new_file, signals, signals_headers, header, digital=True)

    # Safety check: are we able to load the new EDF file?
    # Get new EDF start, stop and duration
    with pyedflib.EdfReader(new_file) as edf:
        start = edf.getStartdatetime()
        stop = start + timedelta(seconds=edf.getFileDuration())
        duration = stop - start
        edf.close()

    # Verbose
    if verbose:
        print(f"Original: {current_start} to {current_stop} ({current_duration})")
        print(f"Truncated: {start} to {stop} ({duration})")
        print(f"Succesfully written file: {new_file}")


def rename_channels(edf_file, mapping, new_file=None, verbose=False):
    """
    A convenience function to rename channels in an EDF file.

    Parameters
    ----------
    edf_file : str
        an string pointing to an edf file.
    mapping : dict
         a dictionary with channel mappings as key:value.
         eg: {'M1-O2':'A1-O2'}
    new_file : str, optional
        the new filename. If None will be edf_file + '_renamed'
        The default is None.
    verbose : bool, optional
        print progress or not. The default is False.

    Returns
    -------
    bool
        True if successful, False if failed.

    """
    header = read_edf_header(edf_file)
    channels = header['channels']
    if new_file is None:
        file, ext = os.path.splitext(edf_file)
        new_file = file + '_renamed' + ext

    signal_headers = []
    signals = []
    for ch_nr in tqdm(range(len(channels)), disable=not verbose):
        signal, signal_header, _ = read_edf(edf_file, digital=True,
                                            ch_nrs=ch_nr, verbose=verbose)
        ch = signal_header[0]['label']
        if ch in mapping :
            if verbose: print(f'{ch} to {mapping[ch]}')
            ch = mapping[ch]
            signal_header[0]['label']=ch
        else:
            if verbose: print(f'no mapping for {ch}, leave as it is')
        signal_headers.append(signal_header[0])
        signals.append(signal.squeeze())

    return write_edf(new_file, signals, signal_headers, header, digital=True)


def change_polarity(edf_file, channels, new_file=None, verify=True,
                    verbose=False):
    """
    Change polarity of certain channels

    Parameters
    ----------
    edf_file : str
        from which file to change polarity.
    channels : list of int
        the indices of the channels.
    new_file : str, optional
        where to save the edf with inverted channels. The default is None.
    verify : bool, optional
        whether to verify the two edfs for similarity. The default is True.
    verbose : str, optional
        print progress or not. The default is True.

    Returns
    -------
    bool
        True if success.

    """

    if new_file is None:
        new_file = os.path.splitext(edf_file)[0] + '.edf'

    if isinstance(channels, str): channels=[channels]
    channels = [c.lower() for c in channels]

    signals, signal_headers, header = read_edf(edf_file, digital=True,
                                               verbose=verbose)
    for i,sig in enumerate(signals):
        label = signal_headers[i]['label'].lower()
        if label in channels:
            if verbose: print(f'inverting {label}')
            signals[i] = -sig
    write_edf(new_file, signals, signal_headers, header,
              digital=True, correct=False, verbose=verbose)
    if verify: compare_edf(edf_file, new_file)
    return True
