# -*- coding: utf-8 -*-
# Copyright (c) 2015 Holger Nahrstaedt
from __future__ import division, print_function, absolute_import

import os
import numpy as np
from numpy.testing import (assert_raises, run_module_suite,
                           assert_equal, assert_allclose, assert_almost_equal)

import pyedflib

data_dir = os.path.join(os.path.dirname(__file__), 'data')
edf_data_file = os.path.join(data_dir, 'test_generator.edf')


def test_EdfReader():
    f = pyedflib.edfreader.EdfReader(edf_data_file)
    ann_index, ann_duration, ann_text = f.readAnnotations()
    assert_almost_equal(ann_index[0], 0)
    assert_almost_equal(ann_index[1], 600)

    assert_equal(f.signals_in_file, 11)
    assert_equal(f.datarecords_in_file, 600)
    assert_equal(f.getSignalTextLabels()[0], b'squarewave')
    for i in np.arange(11):
        assert_almost_equal(f.getSignalFreqs()[i], 200)
        assert_equal(f.getNSamples()[i], 120000)

    assert_equal(f.getSignalTextLabels()[1].rstrip(), b'ramp')
    assert_equal(f.getSignalTextLabels()[2].rstrip(), b'pulse')
    assert_equal(f.getSignalTextLabels()[3].rstrip(), b'noise')
    assert_equal(f.getSignalTextLabels()[4].rstrip(), b'sine 1 Hz')
    assert_equal(f.getSignalTextLabels()[5].rstrip(), b'sine 8 Hz')
    f._close()
    del f


if __name__ == '__main__':
    run_module_suite()