#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

import numpy as np

import pyedflib


from stacklineplot import stackplot


if __name__ == '__main__':
    f = pyedflib.EdfReader("data/test_generator.edf")
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()

    sigbufs = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
        sigbufs[i, :] = f.readSignal(i)

    f._close()
    del f
    # stackplot(sigbufs,seconds=10.0, start_time=0.0, ylabels=signal_labels)

    stackplot(sigbufs[:, :2000], ylabels=signal_labels)
