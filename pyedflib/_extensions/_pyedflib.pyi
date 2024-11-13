from typing import List, Dict, Union, Optional
import numpy as np
from datetime import datetime

__all__ = ['lib_version', 'CyEdfReader', 'set_patientcode', 'set_starttime_subsecond',
           'write_annotation_latin1', 'write_annotation_utf8', 'set_technician', 'EdfAnnotation',
           'get_annotation', 'read_int_samples', 'blockwrite_digital_samples', 'blockwrite_physical_samples',
           'set_recording_additional', 'write_physical_samples', 'set_patientname', 'set_physical_minimum',
           'read_physical_samples', 'close_file', 'set_physical_maximum', 'open_file_writeonly',
           'set_patient_additional', 'set_digital_maximum', 'set_birthdate', 'set_digital_minimum',
           'write_digital_samples', 'set_equipment', 'set_samples_per_record', 'set_admincode', 'set_label',
           'tell', 'rewind', 'set_sex', 'set_gender', 'set_physical_dimension', 'set_transducer',
           'set_prefilter', 'seek', 'set_startdatetime', 'set_datarecord_duration',
           'set_number_of_annotation_signals', 'open_errors', 'FILETYPE_EDFPLUS',
           'FILETYPE_EDF', 'FILETYPE_BDF', 'FILETYPE_BDFPLUS', 'write_errors', 'get_number_of_open_files',
           'get_handle', 'is_file_used', 'blockwrite_digital_short_samples', 'write_digital_short_samples']

open_errors: Dict[int, str]
write_errors: Dict[int, str]

FILETYPE_EDF: int
FILETYPE_EDFPLUS: int
FILETYPE_BDF: int
FILETYPE_BDFPLUS: int

def contains_unicode(string: str) -> bool: ...

def get_short_path_name(long_name: str) -> str: ...

def lib_version() -> str: ...

class CyEdfReader:
    def __init__(self, file_name: str, annotations_mode: int = ..., check_file_size: int = ...): ...
    def __dealloc__(self) -> None: ...
    def check_open_ok(self, result: int) -> bool: ...
    def make_buffer(self) -> np.ndarray: ...
    def open(self, file_name: str, annotations_mode: int = ..., check_file_size: int = ...) -> bool: ...
    def read_annotation(self) -> List[List[str]]: ...
    def _close(self) -> None: ...
    def read_digital_signal(self, signalnum: int, start: int, n: int, sigbuf: np.ndarray[np.int32_t]) -> None: ...
    def readsignal(self, signalnum: int, start: int, n: int, sigbuf: np.ndarray[np.float64_t]) -> None: ...
    def load_datarecord(self, db: np.ndarray[np.float64_t], n: int = ...) -> None: ...

    @property
    def file_name(self) -> str: ...

    @file_name.setter
    def file_name(self, value: str) -> None: ...

    @property
    def handle(self) -> int: ...
    @property
    def datarecords_in_file(self) -> int: ...
    @property
    def signals_in_file(self) -> int: ...
    @property
    def file_duration(self) -> float: ...
    @property
    def filetype(self) -> int: ...
    @property
    def patient(self) -> str: ...
    @property
    def recording(self) -> str: ...
    @property
    def datarecord_duration(self) -> float: ...
    @property
    def annotations_in_file(self) -> int: ...
    @property
    def patientcode(self) -> str: ...
    @property
    def sex(self) -> str: ...
    @property
    def gender(self) -> str: ...
    @property
    def birthdate(self) -> str: ...
    @property
    def patientname(self) -> str: ...
    @property
    def patient_additional(self) -> str: ...
    @property
    def startdate_year(self) -> int: ...
    @property
    def startdate_month(self) -> int: ...
    @property
    def startdate_day(self) -> int: ...
    @property
    def starttime_hour(self) -> int: ...
    @property
    def starttime_minute(self) -> int: ...
    @property
    def starttime_second(self) -> int: ...
    @property
    def starttime_subsecond(self) -> int: ...
    @property
    def admincode(self) -> str: ...
    @property
    def technician(self) -> str: ...
    @property
    def equipment(self) -> str: ...
    @property
    def recording_additional(self) -> str: ...

    def signal_label(self, channel: int) -> str: ...
    def samples_in_file(self, channel: int) -> int: ...
    def samples_in_datarecord(self, channel: int) -> int: ...
    def physical_dimension(self, channel: int) -> str: ...
    def physical_max(self, channel: int) -> float: ...
    def physical_min(self, channel: int) -> float: ...
    def digital_max(self, channel: int) -> int: ...
    def digital_min(self, channel: int) -> int: ...
    def prefilter(self, channel: int) -> str: ...
    def transducer(self, channel: int) -> str: ...
    def samplefrequency(self, channel: int) -> float: ...
    def smp_per_record(self, channel: int) -> int: ...

class EdfAnnotation:
    onset: int
    duration: int
    annotation: str

def set_patientcode(handle: int, patientcode: Union[str, bytes]) -> int: ...
def write_annotation_latin1(handle: int, onset: int, duration: int, description: Union[str, bytes]) -> int: ...
def write_annotation_utf8(handle: int, onset: int, duration: int, description: Union[str, bytes]) -> int: ...
def set_technician(handle: int, technician: Union[str, bytes]) -> int: ...
def get_annotation(handle: int, n: int, edf_annotation: EdfAnnotation) -> int: ...
def read_int_samples(handle: int, edfsignal: int, n: int, buf: np.ndarray[np.int32_t]) -> int: ...
def blockwrite_digital_samples(handle: int, buf: np.ndarray[np.int32_t]) -> int: ...
def blockwrite_digital_short_samples(handle: int, buf: np.ndarray[np.int16_t]) -> int: ...
def blockwrite_physical_samples(handle: int, buf: np.ndarray[np.float64_t]) -> int: ...
def set_recording_additional(handle: int, recording_additional: Union[str, bytes]) -> int: ...
def write_digital_short_samples(handle: int, buf: np.ndarray[np.int16_t]) -> int: ...
def write_physical_samples(handle: int, buf: np.ndarray[np.float64_t]) -> int: ...
def set_patientname(handle: int, name: Union[str,bytes]) -> int: ...
def set_physical_minimum(handle: int, edfsignal: int, phys_min: float) -> int: ...
def read_physical_samples(handle: int, edfsignal: int, n: int, buf: np.ndarray[np.float64_t]) -> int: ...
def close_file(handle: int) -> int: ...
def get_number_of_open_files() -> int: ...
def get_handle(file_number: int) -> int: ...
def is_file_used(path: str) -> bool: ...
def set_physical_maximum(handle: int, edfsignal: int, phys_max: float) -> int: ...
def open_file_writeonly(path: str, filetype: int, number_of_signals: int) -> int: ...
def set_patient_additional(handle: int, patient_additional: Union[str, bytes]) -> int: ...
def set_digital_maximum(handle: int, edfsignal: int, dig_max: int) -> int: ...
def set_birthdate(handle: int, birthdate_year: int, birthdate_month: int, birthdate_day: int) -> int: ...
def set_digital_minimum(handle: int, edfsignal: int, dig_min: int) -> int: ...
def write_digital_samples(handle: int, buf: np.ndarray[np.int32_t]) -> int: ...
def set_equipment(handle: int, equipment: Union[str, bytes]) -> int: ...
def set_samples_per_record(handle: int, edfsignal: int, smp_per_record: int) -> int: ...
def set_admincode(handle: int, admincode: Union[str, bytes]) -> int: ...
def set_label(handle: int, edfsignal: int, label: Union[str, bytes]) -> int: ...
def tell(handle: int, edfsignal: int) -> int: ...
def rewind(handle: int, edfsignal: int) -> None: ...
def set_sex(handle: int, sex: Optional[int]) -> int: ...
def set_gender(handle: int, gender: int) -> int: ...
def set_physical_dimension(handle: int, edfsignal: int, phys_dim: Union[str, bytes]) -> int: ...
def set_transducer(handle: int, edfsignal: int, transducer: Union[str, bytes]) -> int: ...
def set_prefilter(handle: int, edfsignal: int, prefilter: Union[str, bytes]) -> int: ...
def seek(handle: int, edfsignal: int, offset: int, whence: int) -> int: ...
def set_startdatetime(handle: int, startdate_year: int, startdate_month: int, startdate_day: int,
                      starttime_hour: int, starttime_minute: int, starttime_second: int) -> int: ...
def set_starttime_subsecond(handle: int, subsecond: int) -> int: ...
def set_datarecord_duration(handle: int, duration: Union[int, float]) -> int: ...
def set_number_of_annotation_signals(handle: int, annot_signals: int) -> int: ...
