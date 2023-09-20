#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import pyedflib
from stacklineplot import stackplot_t

def animate(frame):
    global offset
    for i in np.arange(n):
        sigbufs_plot[i,:] = sigbufs[i][offset:n_plot + offset]
    ax1.clear()
    stackplot_t(np.transpose(sigbufs_plot[:, :n_plot]), ylabels=signal_labels, ax=ax1)
    offset += dt

if __name__ == '__main__':
    f = pyedflib.data.test_generator()
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    n_min = f.getNSamples()[0]
    sigbufs = [np.zeros(f.getNSamples()[i]) for i in np.arange(n)]
    for i in np.arange(n):
        sigbufs[i] = f.readSignal(i)
        if n_min < len(sigbufs[i]):
            n_min = len(sigbufs[i])

    duration = f.getFileDuration()
    f._close()
    del f
    dt = int(duration/5)
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    n_plot = np.min((n_min, 2000))
    sigbufs_plot = np.zeros((n, n_plot))
    offset = 0
    ani = animation.FuncAnimation(fig, animate, interval=200)
    plt.show()
