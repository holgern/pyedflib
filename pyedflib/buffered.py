"""Buffered EDF/BDF writing for real-time (streaming) acquisition.

Real-time acquisition loops usually read the device in small blocks (for example
10-100 samples). Passing such sub-record blocks directly to
:meth:`pyedflib.EdfWriter.writeSamples` makes each call commit a full data record
padded to length, which silently inflates the file duration and distorts the
stored signal. :class:`BufferedEdfWriter` wraps an ``EdfWriter`` and accepts
blocks of arbitrary size, committing a record only once a full record is
available and padding just the final record on :meth:`close`.
"""

import numpy as np

__all__ = ["BufferedEdfWriter"]


class BufferedEdfWriter:
    """Accept arbitrary-size sample blocks and commit only complete data records.

    Parameters
    ----------
    edf_writer : pyedflib.EdfWriter
        A writer whose signal headers and data-record duration are already set.

    Examples
    --------
    >>> writer = EdfWriter("out.edf", 1)
    >>> writer.setSignalHeader(0, header)
    >>> writer.setDatarecordDuration(1.0)
    >>> bw = BufferedEdfWriter(writer)
    >>> while acquiring:
    ...     bw.write_samples([device.read(100)])   # any block size is fine
    >>> bw.close()
    """

    def __init__(self, edf_writer):
        self._w = edf_writer
        self._n = len(edf_writer.channels)
        self._rec = [int(edf_writer.get_smp_per_record(i)) for i in range(self._n)]
        self._buf = [np.array([], dtype=np.float64) for _ in range(self._n)]

    def write_samples(self, data_list):
        """Append one block per signal; flush every complete data record."""
        if len(data_list) != self._n:
            raise ValueError(
                "expected {} signals, got {}".format(self._n, len(data_list))
            )
        for i in range(self._n):
            self._buf[i] = np.concatenate(
                [self._buf[i], np.asarray(data_list[i], dtype=np.float64)]
            )
        while all(len(self._buf[i]) >= self._rec[i] for i in range(self._n)):
            record = [
                np.ascontiguousarray(self._buf[i][: self._rec[i]])
                for i in range(self._n)
            ]
            self._w.writeSamples(record)
            for i in range(self._n):
                self._buf[i] = self._buf[i][self._rec[i]:]

    def flush(self, pad_with_last=True):
        """Write any pending samples, padding the final record.

        The final record is padded with the last acquired value (``pad_with_last``,
        the default) rather than zero, so that no step discontinuity is introduced.
        """
        if all(len(self._buf[i]) == 0 for i in range(self._n)):
            return
        record = []
        for i in range(self._n):
            b = self._buf[i]
            if len(b) < self._rec[i]:
                pad_val = b[-1] if (len(b) and pad_with_last) else 0.0
                b = np.concatenate([b, np.full(self._rec[i] - len(b), pad_val)])
            record.append(np.ascontiguousarray(b[: self._rec[i]]))
            self._buf[i] = np.array([], dtype=np.float64)
        self._w.writeSamples(record)

    def close(self):
        """Flush the remainder and close the underlying writer."""
        self.flush()
        self._w.close()
