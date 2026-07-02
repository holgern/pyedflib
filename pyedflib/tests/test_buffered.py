import numpy as np
import pyedflib
from pyedflib.buffered import BufferedEdfWriter


def _header(fs):
    return dict(
        label="EMG", dimension="mV", sample_frequency=fs,
        physical_min=-5, physical_max=5, digital_min=-32768, digital_max=32767,
        transducer="", prefilter="",
    )


def test_buffered_writer_prevents_inflation(tmp_path):
    fs = 1000
    x = np.random.default_rng(0).standard_normal(10 * fs) * 0.1
    p = str(tmp_path / "buffered.edf")

    w = pyedflib.EdfWriter(p, 1)
    w.setSignalHeader(0, _header(fs))
    w.setDatarecordDuration(1.0)
    bw = BufferedEdfWriter(w)
    for s in range(0, len(x), 100):        # 100-sample streaming blocks
        bw.write_samples([x[s:s + 100]])
    bw.close()

    r = pyedflib.EdfReader(p)
    duration = r.file_duration
    back = r.readSignal(0)
    r.close()

    assert abs(duration - 10.0) < 1e-6      # no inflation
    rms_ratio = np.sqrt(np.mean(back ** 2)) / np.sqrt(np.mean(x ** 2))
    assert abs(rms_ratio - 1.0) < 0.02      # amplitude preserved


def test_naive_subrecord_blocks_inflate_file(tmp_path):
    """Documents the behaviour that BufferedEdfWriter avoids."""
    fs = 1000
    x = np.random.default_rng(0).standard_normal(10 * fs) * 0.1
    p = str(tmp_path / "naive.edf")

    w = pyedflib.EdfWriter(p, 1)
    w.setSignalHeader(0, _header(fs))
    w.setDatarecordDuration(1.0)
    for s in range(0, len(x), 100):
        w.writeSamples([np.ascontiguousarray(x[s:s + 100])])
    w.close()

    r = pyedflib.EdfReader(p)
    duration = r.file_duration
    r.close()

    assert duration == 100.0                # 10x inflation without buffering


def test_buffered_writer_multichannel(tmp_path):
    fs = 1000
    rng = np.random.default_rng(1)
    x = [rng.standard_normal(10 * fs) * 0.1 for _ in range(2)]
    p = str(tmp_path / "buffered_mc.edf")

    w = pyedflib.EdfWriter(p, 2)
    w.setSignalHeaders([_header(fs), _header(fs)])
    w.setDatarecordDuration(1.0)
    bw = BufferedEdfWriter(w)
    for s in range(0, len(x[0]), 100):
        bw.write_samples([x[0][s:s + 100], x[1][s:s + 100]])
    bw.close()

    r = pyedflib.EdfReader(p)
    assert abs(r.file_duration - 10.0) < 1e-6
    r.close()
