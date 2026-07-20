#!/usr/bin/env python

import numpy as np

import pyedflib

if __name__ == '__main__':
    f = pyedflib.data.test_generator()
    print(f"\nlibrary version: {pyedflib.version.version}")

    print("\ngeneral header:\n")

    # print(f"filetype: {hdr.filetype}\n")
    print(f"edfsignals: {f.signals_in_file}")
    print(f"file duration: {f.file_duration} seconds")
    print(f"startdate: {f.getStartdatetime().day}-{f.getStartdatetime().month}-{f.getStartdatetime().year}")
    print(f"starttime: {f.getStartdatetime().hour}:{f.getStartdatetime().minute:02}:{f.getStartdatetime().second:02}")
    # print(f"patient: {f.getP}")
    # print(f"recording: {f.getPatientAdditional()}")
    print(f"patientcode: {f.getPatientCode()}")
    print(f"gender: {f.getGender()}")
    print(f"birthdate: {f.getBirthdate()}")
    print(f"patient_name: {f.getPatientName()}")
    print(f"patient_additional: {f.getPatientAdditional()}")
    print(f"admincode: {f.getAdmincode()}")
    print(f"technician: {f.getTechnician()}")
    print(f"equipment: {f.getEquipment()}")
    print(f"recording_additional: {f.getRecordingAdditional()}")
    print(f"datarecord duration: {f.getFileDuration():f} seconds")
    print(f"number of datarecords in the file: {f.datarecords_in_file}")
    print(f"number of annotations in the file: {f.annotations_in_file}")

    channel = 3
    print(f"\nsignal parameters for the {channel}.channel:\n\n")

    print(f"label: {f.getLabel(channel)}")
    print(f"samples in file: {f.getNSamples()[channel]}")
    # print(f"samples in datarecord: {f.get}")
    print(f"physical maximum: {f.getPhysicalMaximum(channel):f}")
    print(f"physical minimum: {f.getPhysicalMinimum(channel):f}")
    print(f"digital maximum: {f.getDigitalMaximum(channel)}")
    print(f"digital minimum: {f.getDigitalMinimum(channel)}")
    print(f"physical dimension: {f.getPhysicalDimension(channel)}")
    print(f"prefilter: {f.getPrefilter(channel)}")
    print(f"transducer: {f.getTransducer(channel)}")
    print(f"samplefrequency: {f.getSampleFrequency(channel):f}")

    annotations = f.readAnnotations()
    for n in np.arange(f.annotations_in_file):
        print(f"annotation: onset is {annotations[0][n]:f}    duration is {annotations[1][n]}    description is {annotations[2][n]}")

    buf = f.readSignal(channel)
    n = 200
    print(f"\nread {n} samples\n")
    result = ""
    for i in np.arange(n):
        result += (f"{buf[i]:.1f}, ")
    print(result)
    f._close()
    del f
