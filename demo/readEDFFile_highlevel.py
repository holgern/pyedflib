#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

import pyedflib
import json


if __name__ == '__main__':
    edf_file = pyedflib.data.get_generator_filename()
    signals, signal_headers, header = pyedflib.highlevel.read_edf(edf_file)
    print(header)
    print(signal_headers[0])
    print("\nlibrary version: %s" % pyedflib.version.version)

    print("\ngeneral header:\n")

    # print("filetype: %i\n"%hdr.filetype);
    print("edfsignals: %i" % len(signals))
    print("file duration: %i seconds" % (len(signals[0])/signal_headers[0]["sample_rate"]))
    print(json.dumps(header, indent=4, default=str))
    print("number of annotations in the file: %i" % len(header["annotations"]))

    channel = 3
    print("\nsignal parameters for the %d.channel:\n\n" % channel)

    print(json.dumps(signal_headers[channel], indent=4))
    print("samples in file: %i" % len(signals[channel]))

    for ann in header["annotations"]:
        print("annotation: onset is %f    duration is %s    description is %s" % (ann[0],ann[1],ann[2]))

    n = 200
    print("\nread %i samples\n" % n)
    result = signals[channel][:n]
    print(result)
