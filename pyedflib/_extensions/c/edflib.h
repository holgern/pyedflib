/*
*****************************************************************************
*
* Copyright (c) 2009 - 2024 Teunis van Beelen
* All rights reserved.
*
* Email: teuniz@protonmail.com
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of the copyright holder nor the names of its
*       contributors may be used to endorse or promote products derived from
*       this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ''AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
*****************************************************************************
*/

/**
 * @file edflib.h
 *
 * In EDF, the resolution (or sensitivity) (e.g. uV/bit) and offset are stored using four parameters:<br>
 * digital maximum and minimum, and physical maximum and minimum.<br>
 * Here, digital means the raw data coming from a sensor or ADC. Physical means the units like uV.<br>
 * The resolution in units per least significant bit is calculated as follows:<br>
 *
 * units per bit = (physical max - physical min) / (digital max - digital min)<br>
 *
 * The digital offset is calculated as follows:<br>
 *
 * offset = (physical max / units per bit) - digital max<br>
 *
 * For a better explanation about the relation between digital data and physical data,<br>
 * read the document "Coding Schemes Used with Data Converters" (PDF):<br>
 *
 * https://www.ti.com/general/docs/lit/getliterature.tsp?baseLiteratureNumber=sbaa042<br>
 *
 * An EDF file usually contains multiple so-called datarecords. One datarecord usually has a duration of one second (this is the default but it's not mandatory).<br>
 * In that case a file with a duration of five minutes contains 300 datarecords. The duration of a datarecord can be freely chosen but, if possible, use values from<br>
 * 0.1 to 1 second for easier handling. Just make sure that the total size of one datarecord, expressed in bytes, does not exceed 10 MByte (15 MBytes for BDF(+)).<br>
 *
 * The <b>recommendation</b> of a maximum datarecord size of 61440 bytes in the EDF(+) specification was useful in the time people were still using DOS as their main operating system.<br>
 * Using DOS and fast (near) pointers (16-bit pointers), the maximum allocatable block of memory was 64KByte.<br>
 * This is not a concern anymore so the maximum datarecord size now is limited to 10 MByte for EDF(+) and 15 MByte for BDF(+). This helps to accommodate for higher sampling rates<br>
 * used by modern Analog to Digital Converters.<br>
 *
 * EDF header character encoding: The EDF specification says that only (printable) ASCII characters are allowed.<br>
 * When writing the header info, EDFlib will assume you are using Latin1 encoding and it will automatically convert<br>
 * characters with accents, umlauts, tilde, etc. to their "normal" equivalent without the accent/umlaut/tilde/etc.<br>
 * in order to create a valid EDF file.<br>
 * The description of an EDF+ annotation/event/trigger on the other hand, is always encoded in UTF-8 (which is forward compatible with ASCII).<br>
 *
 * The sample frequency of a signal is calculated as follows: sf = (smp_in_datarecord * EDFLIB_TIME_DIMENSION) / datarecord_duration<br>
 *
 * Annotation signals<br>
 * ==================<br>
 *
 * EDF+ and BDF+ store the annotations/events/triggers in one or more signals (in order to be backwards compatible with EDF and BDF)<br>
 * and they can appear anywhere in the list of signals.<br>
 * The numbering of the signals in the file is zero based (starts at 0). Signals used for annotations are skipped by EDFlib.<br>
 * This means that the annotationsignal(s) in the file are hidden.<br>
 * Use the function edf_get_annotation() to get the annotations.<br>
 *
 * So, when a file contains 7 signals and the third and fifth signal are annotation signals, the library will<br>
 * report that there are only 5 signals in the file.<br>
 * The library will "map" the (zero-based) signal numbers as follows: 0->0, 1->1, 2->3, 3->4, 4->6.<br>
 * This way you don't need to worry about which signals are annotationsignals, the library will take care of it.<br>
 *
 * How the library stores time values<br>
 * ==================================<br>
 *
 * To avoid rounding errors and to be able to compare values, the library stores some time values in variables of type long long int.<br>
 * In order not to lose the sub-second precision, all time values are scaled with a scaling factor: 10000000.<br>
 * This will limit the time resolution to 100 nanoseconds. To calculate the amount of seconds, divide<br>
 * the timevalue by 10000000 or use the macro EDFLIB_TIME_DIMENSION which is declared in edflib.h.<br>
 * The following variables use this scaling when you open a file in read mode: "file_duration", "starttime_subsecond" and "onset".<br>
 *
 * EDFlib and thread-safety<br>
 * ========================<br>
 * The following functions are always MT-unsafe:<br>
 * edfopen_file_readonly()   (race condition)<br>
 * edfclose_file()           (race condition)<br>
 * edflib_get_handle()       (race condition)<br>
 *
 * When writing to or reading from the same file, all EDFlib functions are MT-unsafe (race condition).<br>
 *
 * When accessing EDFlib from different threads, use a mutex.<br>
 *
 * For more info about the EDF and EDF+ format, visit: https://edfplus.info/specs/<br>
 *
 * For more info about the BDF and BDF+ format, visit: https://www.teuniz.net/edfbrowser/bdfplus%20format%20description.html<br>
 *
 */


/* compile with options "-D_LARGEFILE64_SOURCE -D_LARGEFILE_SOURCE" */


#ifndef EDFLIB_INCLUDED
#define EDFLIB_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


/*
 * If both EDFLIB_SO_DLL and EDFLIB_BUILD are defined: compile only EDFlib as a shared library (so, dll).
 * When compiling on unix-like systems, add the -fvisibility=hidden to hide all symbols by
 * default so that this macro can reveal them.
 *
 * If only EDFLIB_SO_DLL is defined: link with EDFlib as an external library (so, dll).
 * EDFlib must be installed on your system (as an .so or .dll) when running your program.
 *
 * If both EDFLIB_SO_DLL and EDFLIB_BUILD are not defined: EDFlib will not be used as a shared library,
 * it will be an integral part of your program instead.
 *
 */

/*
#define EDFLIB_SO_DLL
#define EDFLIB_BUILD
*/

#if defined(EDFLIB_SO_DLL)
#  if defined(EDFLIB_BUILD)
#    if defined(_WIN32)
#      define EDFLIB_API __declspec(dllexport)
#    elif defined(__ELF__)
#      define EDFLIB_API __attribute__ ((visibility ("default")))
#    else
#      define EDFLIB_API
#    endif
#  else
#    if defined(_WIN32)
#      define EDFLIB_API __declspec(dllimport)
#    else
#      define EDFLIB_API
#    endif
#  endif
#else
#  define EDFLIB_API
#endif



#define EDFLIB_TIME_DIMENSION     (10000000LL)
#define EDFLIB_MAXSIGNALS                (640)
#define EDFLIB_MAX_ANNOTATION_LEN        (512)

#define EDFSEEK_SET  (0)
#define EDFSEEK_CUR  (1)
#define EDFSEEK_END  (2)

#define EDF_ANNOT_IDX_POS_END     (0)
#define EDF_ANNOT_IDX_POS_MIDDLE  (1)
#define EDF_ANNOT_IDX_POS_START   (2)

/* the following defines are used in the member "filetype" of the edf_hdr_struct
   and as return value for the function edfopen_file_readonly() */
#define EDFLIB_FILETYPE_EDF                  (0)
#define EDFLIB_FILETYPE_EDFPLUS              (1)
#define EDFLIB_FILETYPE_BDF                  (2)
#define EDFLIB_FILETYPE_BDFPLUS              (3)
#define EDFLIB_MALLOC_ERROR                 (-1)
#define EDFLIB_NO_SUCH_FILE_OR_DIRECTORY    (-2)

/* when this error occurs, try to open the file with EDFbrowser (https://www.teuniz.net/edfbrowser/),
   it will give you full details about the cause of the error. It can also fix most errors. */
#define EDFLIB_FILE_CONTAINS_FORMAT_ERRORS  (-3)

#define EDFLIB_MAXFILES_REACHED             (-4)
#define EDFLIB_FILE_READ_ERROR              (-5)
#define EDFLIB_FILE_ALREADY_OPENED          (-6)
#define EDFLIB_FILETYPE_ERROR               (-7)
#define EDFLIB_FILE_WRITE_ERROR             (-8)
#define EDFLIB_NUMBER_OF_SIGNALS_INVALID    (-9)
#define EDFLIB_FILE_IS_DISCONTINUOUS       (-10)
#define EDFLIB_INVALID_READ_ANNOTS_VALUE   (-11)
#define EDFLIB_ARCH_ERROR                  (-12)

/* values for annotations */
#define EDFLIB_DO_NOT_READ_ANNOTATIONS  (0)
#define EDFLIB_READ_ANNOTATIONS         (1)
#define EDFLIB_READ_ALL_ANNOTATIONS     (2)

/* the following defines are possible errors returned by the first sample write action */
#define EDFLIB_NO_SIGNALS                  (-20)
#define EDFLIB_TOO_MANY_SIGNALS            (-21)
#define EDFLIB_NO_SAMPLES_IN_RECORD        (-22)
#define EDFLIB_DIGMIN_IS_DIGMAX            (-23)
#define EDFLIB_DIGMAX_LOWER_THAN_DIGMIN    (-24)
#define EDFLIB_PHYSMIN_IS_PHYSMAX          (-25)
#define EDFLIB_DATARECORD_SIZE_TOO_BIG     (-26)

#ifdef __cplusplus
extern "C" {
#endif

/**
 * This structure contains the signal parameters.
 */
typedef struct edf_param_struct
{
  char   label[17];              /*!< Label (name) of the signal, null-terminated string. */
  long long smp_in_file;         /*!< Number of samples in the file. */
  double phys_max;               /*!< Physical maximum, usually the maximum input of the ADC. */
  double phys_min;               /*!< Physical minimum, usually the minimum input of the ADC. */
  int    dig_max;                /*!< Digital maximum, usually the maximum output of the ADC, cannot not be higher than 32767 for EDF or 8388607 for BDF. */
  int    dig_min;                /*!< Digital minimum, usually the minimum output of the ADC, cannot not be lower than -32768 for EDF or -8388608 for BDF. */
  int    smp_in_datarecord;      /*!< Number of samplesin a datarecord, if the datarecord has a duration of one second (default), then it equals the sample rate. */
  char   physdimension[9];       /*!< Physical dimension (unit, e.g. uV, bpm, mA, etc.), null-terminated string. */
  char   prefilter[81];          /*!< Prefilter settings, null-terminated string. */
  char   transducer[81];         /*!< Transducer (sensor), null-terminated string. */
} edflib_param_t;

/**
 * This structure is used for annotations/events/triggers.
 */
typedef struct edf_annotation_struct
{
        long long onset;                                /*!< Onset time of the event, expressed in units of 100 nanoseconds and relative to the start of the recording. */
        long long duration_l;                           /*!< Duration, expressed in units of 100 nanoseconds, if less than zero: unused or not applicable. */
        char duration[20];                              /*!< Duration, expressed in seconds, this is a null-terminated ASCII string. */
        char annotation[EDFLIB_MAX_ANNOTATION_LEN + 1]; /*!< Description of the annotation/event/trigger, this is a null-terminated UTF8 string. */
} edflib_annotation_t;

/**
 * This structure contains the general header info and parameters. It will be filled when calling the function edfopen_file_readonly().
 */
typedef struct edf_hdr_struct
{
  int       handle;               /*!< A handle (identifier) used to distinguish the different files or -1 in case of an error. */
  int       filetype;             /*!< 0: EDF, 1: EDF+, 2: BDF, 3: BDF+, a negative number indicates an error code. */
  int       edfsignals;           /*!< Number of signals in the file, annotation channels are not included. */
  long long file_duration;        /*!< Duration of the file expressed in units of 100 nanoseconds. */
  int       startdate_day;        /*!< Startdate: day: 1 - 31 */
  int       startdate_month;      /*!< Startdate: month: 1 - 12 */
  int       startdate_year;       /*!< Startdate: year: 1985 - 2084 */
  long long starttime_subsecond;  /*!< Starttime subsecond expressed in units of 100 nanoseconds. Is always less than 10000000 (one second). Only used by EDF+ and BDF+. */
  int       starttime_second;     /*!< Starttime: second: 0 - 59 */
  int       starttime_minute;     /*!< Starttime: minute: 0 - 59 */
  int       starttime_hour;       /*!< Starttime: hour: 0 - 23 */
  char      patient[81];          /*!< Null-terminated string, contains patient field of header, is always empty when filetype is EDFPLUS or BDFPLUS. */
  char      recording[81];        /*!< Null-terminated string, contains recording field of header, is always empty when filetype is EDFPLUS or BDFPLUS. */
  char      patientcode[81];      /*!< Null-terminated string, is always empty when filetype is EDF or BDF. */
  char      sex[16];              /*!< Null-terminated string, is always empty when filetype is EDF or BDF. */
#if defined(__GNUC__)
  char      gender[16] __attribute__ ((deprecated ("use sex")));      /*!< Deprecated, use \p sex. */
#else
  char      gender[16];  /*!< Deprecated, use \p sex. */
#endif
  char      birthdate[16];             /*!< Null-terminated string, is always empty when filetype is EDF or BDF. */
  int       birthdate_day;             /*!< Birthdate: day: 1 - 31 (zero in case of EDF or BDF). */
  int       birthdate_month;           /*!< Birthdate: month: 1 - 12 (zero in case of EDF or BDF). */
  int       birthdate_year;            /*!< Birthdate: year: 1800 - 3000 (zero in case of EDF or BDF). */
  char      patient_name[81];          /*!< Null-terminated string, is always empty when filetype is EDF or BDF. */
  char      patient_additional[81];    /*!< Null-terminated string, is always empty when filetype is EDF or BDF. */
  char      admincode[81];             /*!< Null-terminated string, is always empty when filetype is EDF or BDF. */
  char      technician[81];            /*!< Null-terminated string, is always empty when filetype is EDF or BDF. */
  char      equipment[81];             /*!< Null-terminated string, is always empty when filetype is EDF or BDF. */
  char      recording_additional[81];  /*!< Null-terminated string, is always empty when filetype is EDF or BDF. */
  long long datarecord_duration;       /*!< Duration of a datarecord expressed in units of 100 nanoseconds. */
  long long datarecords_in_file;       /*!< Number of datarecords in the file. */
  long long annotations_in_file;       /*!< Number of annotations/events/triggers in the file. */
  edflib_param_t signalparam[EDFLIB_MAXSIGNALS];  /*!< array of structs containing the signal parameters. */
} edflib_hdr_t;

/*****************  the following functions are used to read files **************************/

/**
 * Opens an existing file for reading.
 *
 * @param[in] path
 * null-terminated string containing the \p path to the file
 *
 * @param[out] edfhdr
 * pointer to an \p edflib_hdr_t struct, all fields in this struct will be overwritten,
 * it will be filled with all the relevant header- and signalinfo/parameters
 *
 * @param[in] read_annotations
 * Must have one of the following values:
 * - EDFLIB_DO_NOT_READ_ANNOTATIONS      annotations will not be read (this can save time when opening a very large EDF+ or BDF+ file
 * - EDFLIB_READ_ANNOTATIONS             annotations will be read immediately, stops when an annotation has
 *                                       been found which contains the description "Recording ends"
 * - EDFLIB_READ_ALL_ANNOTATIONS         all annotations will be read immediately
 *
 * @return
 * 0 on success, in case of an error it returns -1 and an error code will be set in the member "filetype" of edfhdr.
 * This function is required if you want to read a file
 *
 * In case of a file format error (-3), try to open the file with EDFbrowser: https://www.teuniz.net/edfbrowser/
 * It will give you full details about the cause of the error and it can also fix most errors.
 */
EDFLIB_API int edfopen_file_readonly(const char *path, edflib_hdr_t *edfhdr, int read_annotations);

/**
 * Reads \p n samples from \p edfsignal, starting from the current sample position indicator, into \p buf (edfsignal starts at 0).
 * The values are converted to their physical values e.g. microVolts, beats per minute, etc.
 *
 * @param[in] handle
 * File handle.
 * @param[in] edfsignal
 * The zero-based index of the signal.
 * @param[in] n
 * Number of samples to read. The sample position indicator will be increased with the same amount.
 * @param[out] buf
 * Pointer to a buffer, size must be equal to, or bigger than, sizeof(double[n])
 *
 * @return
 * The number of samples read (this can be less than \p n or zero!) or -1 in case of an error
 */
EDFLIB_API int edfread_physical_samples(int handle, int edfsignal, int n, double *buf);

/**
 * Reads \p n samples from \p edfsignal, starting from the current sample position indicator, into \p buf (edfsignal starts at 0).
 * The values are the "raw" digital values (e.g. from an ADC).
 *
 * @param[in] handle
 * File handle.
 * @param[in] edfsignal
 * The zero-based index of the signal.
 * @param[in] n
 * Number of samples to read. The sample position indicator will be increased with the same amount.
 * @param[out] buf
 * Pointer to a buffer, size must be equal to, or bigger than, sizeof(double[n])
 *
 * @return
 * The number of samples read (this can be less than \p n or zero!) or -1 in case of an error
 */
EDFLIB_API int edfread_digital_samples(int handle, int edfsignal, int n, int *buf);

/**
 * Sets the sample position indicator for the edfsignal pointed to by \p edfsignal.
 * The new position, measured in samples, is obtained by adding offset samples to the position specified by \p whence.
 * If \p whence is set to EDFSEEK_SET, EDFSEEK_CUR, or EDFSEEK_END, the offset is relative to the start of the file,
 * the current position indicator, or end-of-file, respectively.
 * Note that every signal has it's own independent sample position indicator and \p edfseek() affects only one of them.
 *
 * @param[in] handle
 * File handle.
 * @param[in] edfsignal
 * The zero-based index of the signal.
 * @param[in] offset
 * Offset measured in samples.
 * @param[in] whence
 * Reference for \p offset:
 * - EDFSEEK_SET start of the file
 * - EDFSEEK_CUR current position
 * - EDFSEEK_END end of the file
 *
 * @return
 * The current offset or -1 in case of an error.
 */
EDFLIB_API long long edfseek(int handle, int edfsignal, long long offset, int whence);

/**
 * Obtains the current value of the sample position indicator for the edfsignal pointed to by \p edfsignal.
 * Note that every signal has it's own independent sample position indicator and \p edftell() affects only one of them.
 *
 * @param[in] handle
 * File handle.
 * @param[in] edfsignal
 * The zero-based index of the signal.
 *
 * @return
 * The current offset or -1 in case of an error.
 */
EDFLIB_API long long edftell(int handle, int edfsignal);

/**
 * Sets the sample position indicator for the edfsignal pointed to by \p edfsignal to the beginning of the file.
 * It is equivalent to: \p edfseek(handle, edfsignal, 0LL, EDFSEEK_SET).
 * Note that every signal has it's own independent sample position indicator and \p edfrewind() affects only one of them.
 *
 * @param[in] handle
 * File handle.
 * @param[in] edfsignal
 * The zero-based index of the signal.
 *
 * @return
 * 0 on success or -1 in case of an error.
 */
EDFLIB_API int edfrewind(int handle, int edfsignal);

/**
 * Fills the edflib_annotation_t structure with the annotation \p n.
 * The string that describes the annotation/event is encoded in UTF-8.
 * To obtain the number of annotations in a file, check edf_hdr_struct -> annotations_in_file.
 *
 * @param[in] handle
 * File handle.
 * @param[in] n
 * The zero-based index number of the list of annotations.
 * @param[out] annot
 * Pointer to a struct that will be filled with the annotation.
 *
 * @return
 * 0 on success or -1 in case of an error.
 */
EDFLIB_API int edf_get_annotation(int handle, int n, edflib_annotation_t *annot);

/*****************  the following functions are used in read and write mode **************************/

/**
 * Closes (and in case of writing, finalizes) the file.
 *
 * This function MUST be called when you have finished reading or writing
 * This function is required after reading or writing. Failing to do so will cause
 * unnecessary memory usage and in case of writing it will cause a corrupted or incomplete file.
 *
 * @param[in] handle
 * File handle.
 *
 * @return
 * 0 on success or -1 in case of an error.
 */
EDFLIB_API int edfclose_file(int handle);

/**
 * Returns the version number of this library, multiplied by hundred. if version is "1.00" then it will return 100.
 *
 * @return
 * The version number.
 */
EDFLIB_API int edflib_version(void);

/**
 * Returns 1 if the file is in use, either for reading or writing, otherwise returns 0.
 *
 * @param[in] path
 * Pointer to a null-terminated string that contains the path to the file.
 *
 * @return
 * 1 if the file is in use (either for reading or writing), otherwise 0.
 */
EDFLIB_API int edflib_is_file_used(const char *path);

/**
 * Returns the number of open files.
 *
 * @return
 * The number of open files, either for reading or writing.
 */
EDFLIB_API int edflib_get_number_of_open_files(void);

/**
 * Returns the handle of an open file, either for reading or writing.
 *
 * @param[in] file_number
 * A zero based index number of the list of open files.
 *
 * @return
 * The file handle or -1 if the file_number >= number of open files.
 */
EDFLIB_API int edflib_get_handle(int file_number);

/*****************  the following functions are used to write files **************************/

/**
 * Opens an new file for writing. Warning: an already existing file with the same name will be silently overwritten without advance warning!<br>
 * This function is required if you want to write a file (or use edfopen_file_writeonly_with_params())
 *
 * @param[in] path
 * A null-terminated string containing the path and name of the file
 *
 * @param[in] filetype
 * Must be EDFLIB_FILETYPE_EDFPLUS or EDFLIB_FILETYPE_BDFPLUS.
 *
 * @param[in] number_of_signals
 * The number of signals you want to store into the file<br>
 * (excluding annotation signals, the library will take care of that).
 *
 * @return
 * A file handle on success or a negative number in case of an error:
 * - EDFLIB_MALLOC_ERROR
 * - EDFLIB_NO_SUCH_FILE_OR_DIRECTORY
 * - EDFLIB_MAXFILES_REACHED
 * - EDFLIB_FILE_ALREADY_OPENED
 * - EDFLIB_NUMBER_OF_SIGNALS_INVALID
 * - EDFLIB_ARCH_ERROR
 */
EDFLIB_API int edfopen_file_writeonly(const char *path, int filetype, int number_of_signals);

/**
 * This is a convenience function that can create a new EDF file and initializes the most important parameters.<br>
 * It assumes that all signals are sharing the same parameters (you can still change them though).<br>
 * Warning: an already existing file with the same name will be silently overwritten without advance warning!<br>
 *
 * @param[in] path
 * A null-terminated string containing the path and name of the file.
 *
 * @param[in] filetype
 * Must be EDFLIB_FILETYPE_EDFPLUS or EDFLIB_FILETYPE_BDFPLUS.
 *
 * @param[in] number_of_signals
 * The number of signals you want to store into the file<br>
 * (excluding annotation signals, the library will take care of that).
 *
 * @param[in] samplefrequency
 * Sample frequency for all signals. (In reality, it sets the number of samples per datarecord which equals the sample frequency only when<br>
 * the datarecords have a duration of one second which is the default here.)
 *
 * @param[in] phys_max_min
 * Physical maximum and minimum for all signals.
 *
 * @param[in] phys_dim
 * Pointer to a NULL-terminated ASCII-string containing the physical dimension (unit) for all signals ("uV", "BPM", "mA", "Degr.", etc.).
 *
 * @return
 * A file handle on success or a negative number in case of an error:
 * - EDFLIB_MALLOC_ERROR
 * - EDFLIB_NO_SUCH_FILE_OR_DIRECTORY
 * - EDFLIB_MAXFILES_REACHED
 * - EDFLIB_FILE_ALREADY_OPENED
 * - EDFLIB_NUMBER_OF_SIGNALS_INVALID
 * - EDFLIB_ARCH_ERROR
 */
EDFLIB_API int edfopen_file_writeonly_with_params(const char *path, int filetype, int number_of_signals, int samplefrequency, double phys_max_min, const char *phys_dim);

/**
 * Sets the sample frequency of signal edfsignal. In reality, it sets the number of samples in a datarecord<br>
 * which equals the sample frequency only when the datarecords have a duration of one second.<br>
 * The effective sample frequency is: samplefrequency / datarecord duration<br>
 * This function is required for every signal (except when using edfopen_file_writeonly_with_params()) and can be called<br>
 * only after opening a file in write mode and before the first sample write action.<br>
 *
 * @param[in] handle
 * File handle.
 *
 * @param[in] edfsignal
 * The zero-based index of the signal.
 *
 * @param[in] samplefrequency
 * Sample frequency, must be > 0;
 *
 * @return
 * 0 on success, otherwise -1.<br>
 */
EDFLIB_API int edf_set_samplefrequency(int handle, int edfsignal, int samplefrequency);

/**
 * Sets the maximum physical value of signal edfsignal. (the value of the input of the ADC when the output equals the value of "digital maximum")<br>
 * It is the highest value that the equipment is able to record. It does not necessarily mean the signal recorded reaches this level.<br>
 * In other words, it is the highest value that CAN occur in the recording.<br>
 * Must be un-equal to physical minimum.<br>
 * This function is required for every signal (except when using edfopen_file_writeonly_with_params()) and can be called<br>
 * only after opening a file in write mode and before the first sample write action.<br>
 *
 * @param[in] handle
 * File handle.
 *
 * @param[in] edfsignal
 * The zero-based index of the signal.
 *
 * @param[in] phys_max
 * Physical maximum, must be != physical minimum;
 *
 * @return
 * 0 on success, otherwise -1.<br>
 */
EDFLIB_API int edf_set_physical_maximum(int handle, int edfsignal, double phys_max);

/**
 * Sets the minimum physical value of signal edfsignal. (the value of the input of the ADC when the output equals the value of "digital minimum")<br>
 * It is the lowest value that the equipment is able to record. It does not necessarily mean the signal recorded reaches this level.<br>
 * In other words, it is the lowest value that CAN occur in the recording.<br>
 * Must be un-equal to physical maximum.<br>
 * This function is required for every signal (except when using edfopen_file_writeonly_with_params()) and can be called<br>
 * only after opening a file in write mode and before the first sample write action.<br>
 *
 * @param[in] handle
 * File handle.
 *
 * @param[in] edfsignal
 * The zero-based index of the signal.
 *
 * @param[in] phys_min
 * Physical minimum, must be != physical maximum;
 *
 * @return
 * 0 on success, otherwise -1.<br>
 */
EDFLIB_API int edf_set_physical_minimum(int handle, int edfsignal, double phys_min);

/**
 * Sets the maximum digital value of signal edfsignal. The maximum value is 32767 for EDF+ and 8388607 for BDF+.<br>
 * It is the highest value that the equipment is able to record. It does not necessarily mean the signal recorded reaches this level.<br>
 * In other words, it is the highest value that CAN occur in the recording.<br>
 * Must be higher than digital minimum.<br>
 * This function is required for every signal (except when using edfopen_file_writeonly_with_params()) and can be called<br>
 * only after opening a file in write mode and before the first sample write action.<br>
 *
 * @param[in] handle
 * File handle.
 *
 * @param[in] edfsignal
 * The zero-based index of the signal.
 *
 * @param[in] dig_max
 * Digital maximum, must be > digital minimum;
 *
 * @return
 * 0 on success, otherwise -1.<br>
 */
EDFLIB_API int edf_set_digital_maximum(int handle, int edfsignal, int dig_max);

/**
 * Sets the minimum digital value of signal edfsignal. The minimum value is -32768 for EDF+ and -8388608 for BDF+.<br>
 * It is the lowest value that the equipment is able to record. It does not necessarily mean the signal recorded reaches this level.<br>
 * In other words, it is the lowest value that CAN occur in the recording.<br>
 * Must be lower than digital maximum.<br>
 * This function is required for every signal (except when using edfopen_file_writeonly_with_params()) and can be called<br>
 * only after opening a file in write mode and before the first sample write action.<br>
 *
 * @param[in] handle
 * File handle.
 *
 * @param[in] edfsignal
 * The zero-based index of the signal.
 *
 * @param[in] dig_min
 * Digital minimum, must be < digital maximum;
 *
 * @return
 * 0 on success, otherwise -1.<br>
 */
EDFLIB_API int edf_set_digital_minimum(int handle, int edfsignal, int dig_min);

/**
 * Sets the label (name) of signal \p edfsignal. ("EEG FP1", "SaO2", etc.).<br>
 * This function is recommended for every signal when you want to write a file<br>
 * and can be called only after opening a file in write mode and before the first sample write action.<br>
 *
 * @param[in] handle
 * File handle.
 *
 * @param[in] edfsignal
 * The zero-based index of the signal.
 *
 * @param[in] label
 * A pointer to a NULL-terminated ASCII-string containing the label (name) of the signal \p edfsignal.
 *
 * @return
 * 0 on success, otherwise -1.<br>
 */
EDFLIB_API int edf_set_label(int handle, int edfsignal, const char *label);

/**
 * Sets the prefilter of signal \p edfsignal e.g. "HP:0.1Hz", "LP:75Hz N:50Hz", etc.<br>
 * This function is optional and can be called only after opening a file in writemode and<br>
 * before the first sample write action.
 *
 * @param[in] handle
 * File handle.
 *
 * @param[in] edfsignal
 * The zero-based index of the signal.
 *
 * @param[in] prefilter
 * A pointer to a NULL-terminated ASCII-string containing the prefilter text of the signal \p edfsignal.
 *
 * @return
 * 0 on success, otherwise -1.<br>
 */
EDFLIB_API int edf_set_prefilter(int handle, int edfsignal, const char *prefilter);

/**
 * Sets the transducer of signal \p edfsignal e.g. "AgAgCl cup electrodes", etc.<br>
 * This function is optional and can be called only after opening a file in writemode and<br>
 * before the first sample write action.
 *
 * @param[in] handle
 * File handle.
 *
 * @param[in] edfsignal
 * The zero-based index of the signal.
 *
 * @param[in] transducer
 * A pointer to a NULL-terminated ASCII-string containing the transducer text of the signal \p edfsignal.
 *
 * @return
 * 0 on success, otherwise -1.<br>
 */
EDFLIB_API int edf_set_transducer(int handle, int edfsignal, const char *transducer);

/**
 * Sets the physical dimension (unit) of signal \p edfsignal. ("uV", "BPM", "mA", "Degr.", etc.).<br>
 * This function is recommended for every signal when you want to write a file<br>
 * and can be called only after opening a file in write mode and before the first sample write action.<br>
 *
 * @param[in] handle
 * File handle.
 *
 * @param[in] edfsignal
 * The zero-based index of the signal.
 *
 * @param[in] phys_dim
 * A pointer to a NULL-terminated ASCII-string containing the physical dimension (unit) of the signal \p edfsignal.
 *
 * @return
 * 0 on success, otherwise -1.<br>
 */
EDFLIB_API int edf_set_physical_dimension(int handle, int edfsignal, const char *phys_dim);

/**
 * Sets the startdate and starttime.<br>
 * If not called, the library will use the system date and time at runtime.<br>
 * This function is optional and can be called only after opening a file in write mode<br>
 * and before the first sample write action.<br>
 * Note: for anonymization purposes, the consensus is to use 1985-01-01 00:00:00 for the startdate and starttime.
 *
 * @param[in] handle
 * File handle.
 *
 * @param[in] startdate_year
 * 1985 - 2084 inclusive
 *
 * @param[in] startdate_month
 * 1 - 12 inclusive
 *
 * @param[in] startdate_day
 * 1 - 31 inclusive
 *
 * @param[in] starttime_hour
 * 0 - 23 inclusive
 *
 * @param[in] starttime_minute
 * 0 - 59 inclusive
 *
 * @param[in] starttime_second
 * 0 - 59 inclusive
 *
 * @return
 * 0 on success, otherwise -1.<br>
 */
EDFLIB_API int edf_set_startdatetime(int handle, int startdate_year, int startdate_month, int startdate_day,
                                     int starttime_hour, int starttime_minute, int starttime_second);

/**
 * Sets the subject name<br>
 * This function is optional and can be called only after opening a file in writemode and<br>
 * before the first sample write action.
 *
 * @param[in] handle
 * File handle.
 *
 * @param[in] patientname
 * A pointer to a NULL-terminated ASCII-string containing the subject name.
 *
 * @return
 * 0 on success, otherwise -1.<br>
 */
EDFLIB_API int edf_set_patientname(int handle, const char *patientname);

/**
 * Sets the subject code<br>
 * This function is optional and can be called only after opening a file in writemode and<br>
 * before the first sample write action.
 *
 * @param[in] handle
 * File handle.
 *
 * @param[in] patientcode
 * A pointer to a NULL-terminated ASCII-string containing the subject code.
 *
 * @return
 * 0 on success, otherwise -1.<br>
 */
EDFLIB_API int edf_set_patientcode(int handle, const char *patientcode);

/**
 * Sets the sex of the subject. 1 is male, 0 is female.<br>
 * This function is optional and can be called only after opening a file in writemode<br>
 * and before the first sample write action.<br>
 *
 * @param[in] handle
 * File handle.
 *
 * @param[in] sex
 * 1: male, 0: female.
 *
 * @return
 * 0 on success, otherwise -1.<br>
 */
EDFLIB_API int edf_set_sex(int handle, int sex);

#if defined(__GNUC__)
EDFLIB_API int edf_set_gender(int handle, int sex) __attribute__ ((deprecated ("use edf_set_sex()")));
#else
EDFLIB_API int edf_set_gender(int handle, int sex);
#endif
/* DEPRECATED!! USE edf_set_sex()
 * Sets the sex. 1 is male, 0 is female.
 * Returns 0 on success, otherwise -1
 * This function is optional and can be called only after opening a file in writemode
 * and before the first sample write action
 */

/**
 * Sets the subject birthdate.<br>
 * This function is optional and can be called only after opening a file in write mode<br>
 * and before the first sample write action.<br>
 *
 * @param[in] handle
 * File handle.
 *
 * @param[in] birthdate_year
 * 1800 - 3000 inclusive
 *
 * @param[in] birthdate_month
 * 1 - 12 inclusive
 *
 * @param[in] birthdate_day
 * 1 - 31 inclusive
 *
 * @return
 * 0 on success, otherwise -1.<br>
 */
EDFLIB_API int edf_set_birthdate(int handle, int birthdate_year, int birthdate_month, int birthdate_day);

/**
 * Sets the additional subject info<br>
 * This function is optional and can be called only after opening a file in writemode and<br>
 * before the first sample write action.
 *
 * @param[in] handle
 * File handle.
 *
 * @param[in] patient_additional
 * A pointer to a NULL-terminated ASCII-string containing the additional subject info.
 *
 * @return
 * 0 on success, otherwise -1.<br>
 */
EDFLIB_API int edf_set_patient_additional(int handle, const char *patient_additional);

/**
 * Sets the administration code<br>
 * This function is optional and can be called only after opening a file in writemode and<br>
 * before the first sample write action.
 *
 * @param[in] handle
 * File handle.
 *
 * @param[in] admincode
 * A pointer to a NULL-terminated ASCII-string containing the administration code.
 *
 * @return
 * 0 on success, otherwise -1.<br>
 */
EDFLIB_API int edf_set_admincode(int handle, const char *admincode);

/**
 * Sets the technicians name or code<br>
 * This function is optional and can be called only after opening a file in writemode and<br>
 * before the first sample write action.
 *
 * @param[in] handle
 * File handle.
 *
 * @param[in] technician
 * A pointer to a NULL-terminated ASCII-string containing the technicians name or code.
 *
 * @return
 * 0 on success, otherwise -1.<br>
 */
EDFLIB_API int edf_set_technician(int handle, const char *technician);

/**
 * Sets the equipment brand and/or model<br>
 * This function is optional and can be called only after opening a file in writemode and<br>
 * before the first sample write action.
 *
 * @param[in] handle
 * File handle.
 *
 * @param[in] equipment
 * A pointer to a NULL-terminated ASCII-string containing the equipment brand and/or model<br>
 * used for the recording.
 *
 * @return
 * 0 on success, otherwise -1.<br>
 */
EDFLIB_API int edf_set_equipment(int handle, const char *equipment);

/**
 * Sets the additional info about the recording.<br>
 * This function is optional and can be called only after opening a file in writemode and<br>
 * before the first sample write action.
 *
 * @param[in] handle
 * File handle.
 *
 * @param[in] recording_additional
 * A pointer to a NULL-terminated ASCII-string containing the additional info about the recording.
 *
 * @return
 * 0 on success, otherwise -1.<br>
 */
EDFLIB_API int edf_set_recording_additional(int handle, const char *recording_additional);

/**
 * Writes n physical samples (uV, mA, Ohm) from \p buf belonging to one signal<br>
 * where n is the samplefrequency of that signal.<br>
 * Actually, n equals the number of samples per datarecord which equals the samplefrequency only<br>
 * when the datarecord duration has the default value of one second!<br>
 * The physical samples will be converted to digital samples using the<br>
 * values of physical maximum, physical minimum, digital maximum and digital minimum.<br>
 * Size of \p buf must be equal to or bigger than sizeof(double[samples per datarecord]).<br>
 * Call this function for every signal in the file. The order is important:<br>
 * When there are 4 signals in the file,  the order of calling this function<br>
 * must be: signal 0, signal 1, signal 2, signal 3, signal 0, signal 1, signal 2, etc.<br>
 *
 * @param[in] handle
 * File handle.
 *
 * @param[in] buf
 * A pointer to a buffer containing the samples.
 *
 * @return
 * 0 on success, otherwise -1.<br>
 */
EDFLIB_API int edfwrite_physical_samples(int handle, double *buf);

/**
 * Writes physical samples (uV, mA, Ohm) from \p buf <br>
 * \p buf must be filled with samples from all signals, starting with n samples of signal 0, n samples of signal 1, n samples of signal 2, etc.<br>
 * where n is the samplefrequency of that signal.<br>
 * Actually, n equals the number of samples per datarecord which equals the samplefrequency only<br>
 * when the datarecord duration has the default value of one second!<br>
 * The physical samples will be converted to digital samples using the<br>
 * values of physical maximum, physical minimum, digital maximum and digital minimum.<br>
 * The number of samples written equals the sum of the samples per datarecord of all signals.
 *
 * @param[in] handle
 * File handle.
 *
 * @param[in] buf
 * A pointer to a buffer containing the samples.
 *
 * @return
 * 0 on success, otherwise -1.<br>
 */
EDFLIB_API int edf_blockwrite_physical_samples(int handle, double *buf);

/**
 * Writes n "raw" digital samples from \p buf belonging to one signal<br>
 * where n is the samplefrequency of that signal.<br>
 * Actually, n equals the number of samples per datarecord which equals the samplefrequency only<br>
 * when the datarecord duration has the default value of one second!<br>
 * Size of \p buf should be equal to or bigger than sizeof(short[samples per datarecord]).<br>
 * Call this function for every signal in the file. The order is important:<br>
 * When there are 4 signals in the file,  the order of calling this function<br>
 * must be: signal 0, signal 1, signal 2, signal 3, signal 0, signal 1, signal 2, etc.<br>
 *
 * @param[in] handle
 * File handle.
 *
 * @param[in] buf
 * A pointer to a buffer containing the samples.
 *
 * @return
 * 0 on success, otherwise -1.<br>
 */
EDFLIB_API int edfwrite_digital_short_samples(int handle, short *buf);

/**
 * Writes n "raw" digital samples from \p buf belonging to one signal<br>
 * where n is the samplefrequency of that signal.<br>
 * Actually, n equals the number of samples per datarecord which equals the samplefrequency only<br>
 * when the datarecord duration has the default value of one second!<br>
 * The 16 (or 24 in case of BDF+) least significant bits of the samples will be written to the<br>
 * file without any conversion.<br>
 * Size of \p buf should be equal to or bigger than sizeof(int[samples per datarecord]).<br>
 * Call this function for every signal in the file. The order is important:<br>
 * When there are 4 signals in the file,  the order of calling this function<br>
 * must be: signal 0, signal 1, signal 2, signal 3, signal 0, signal 1, signal 2, etc.<br>
 *
 * @param[in] handle
 * File handle.
 *
 * @param[in] buf
 * A pointer to a buffer containing the samples.
 *
 * @return
 * 0 on success, otherwise -1.<br>
 */
EDFLIB_API int edfwrite_digital_samples(int handle, int *buf);

/**
 * Writes "raw" digital samples from \p buf <br>
 * \p buf must be filled with samples from all signals, starting with n samples of signal 0, n samples of signal 1, n samples of signal 2, etc.<br>
 * where n is the samplefrequency of that signal.<br>
 * Actually, n equals the number of samples per datarecord which equals the samplefrequency only<br>
 * when the datarecord duration has the default value of one second!<br>
 * One sample equals 3 bytes, order is little endian (least significant byte first).<br>
 * Encoding is second's complement, most significant bit of most significant byte is the sign-bit.<br>
 * Because the size of a 3-byte sample is 24-bit, this function can only be used when writing a BDF+ file.<br>
 * The number of samples written equals the sum of the samples per datarecord of all signals.
 *
 * @param[in] handle
 * File handle.
 *
 * @param[in] buf
 * A pointer to a buffer containing the samples.
 *
 * @return
 * 0 on success, otherwise -1.<br>
 */
EDFLIB_API int edf_blockwrite_digital_3byte_samples(int handle, void *buf);

/**
 * Writes "raw" digital samples from \p buf <br>
 * \p buf must be filled with samples from all signals, starting with n samples of signal 0, n samples of signal 1, n samples of signal 2, etc.<br>
 * where n is the samplefrequency of that signal.<br>
 * Actually, n equals the number of samples per datarecord which equals the samplefrequency only<br>
 * when the datarecord duration has the default value of one second!<br>
 * One sample equals 2 bytes, order is little endian (least significant byte first).<br>
 * Encoding is second's complement, most significant bit of most significant byte is the sign-bit.<br>
 * Because the size of a 2-byte sample is 16-bit, this function can only be used when writing an EDF+ file.<br>
 * The number of samples written equals the sum of the samples per datarecord of all signals.
 *
 * @param[in] handle
 * File handle.
 *
 * @param[in] buf
 * A pointer to a buffer containing the samples.
 *
 * @return
 * 0 on success, otherwise -1.<br>
 */
EDFLIB_API int edf_blockwrite_digital_short_samples(int handle, short *buf);

/**
 * Writes "raw" digital samples from \p buf <br>
 * \p buf must be filled with samples from all signals, starting with n samples of signal 0, n samples of signal 1, n samples of signal 2, etc.<br>
 * where n is the samplefrequency of that signal.<br>
 * Actually, n equals the number of samples per datarecord which equals the samplefrequency only<br>
 * when the datarecord duration has the default value of one second!<br>
 * The 16 (or 24 in case of BDF+) least significant bits of the samples will be written to the<br>
 * file without any conversion.<br>
 * The number of samples written equals the sum of the samples per datarecord of all signals.
 *
 * @param[in] handle
 * File handle.
 *
 * @param[in] buf
 * A pointer to a buffer containing the samples.
 *
 * @return
 * 0 on success, otherwise -1.<br>
 */
EDFLIB_API int edf_blockwrite_digital_samples(int handle, int *buf);

/**
 * Writes an annotation/event to the file.<br>
 * This function is optional and can be called only after opening a file in writemode<br>
 * and before closing the file.<br>
 *
 * @param[in] handle
 * File handle.
 *
 * @param[in] onset
 * microseconds since start of recording.
 *
 * @param[in] duration
 * microseconds, > 0 or -1 if not used.
 *
 * @param[in] description
 * A null-terminated UTF8-string containing the text that describes the event.
 *
 * @return
 * 0 on success, otherwise -1.<br>
 */
EDFLIB_API int edfwrite_annotation_utf8_hr(int handle, long long onset, long long duration, const char *description);

#if defined(__GNUC__)
EDFLIB_API int edfwrite_annotation_utf8(int handle, long long onset, long long duration, const char *description) __attribute__ ((deprecated ("use edfwrite_annotation_utf8_hr()")));
#else
EDFLIB_API int edfwrite_annotation_utf8(int handle, long long onset, long long duration, const char *description);
#endif
/* DEPRECATED!! USE edfwrite_annotation_utf8_hr()
 * writes an annotation/event to the file
 * onset is relative to the start of the file
 * onset and duration are in units of 100 microseconds!     resolution is 0.0001 second!
 * for example: 34.071 seconds must be written as 340710
 * if duration is unknown or not applicable: set a negative number (-1)
 * description is a null-terminated UTF8-string containing the text that describes the event
 * This function is optional and can be called only after opening a file in writemode
 * and before closing the file
 */

/**
 * Writes an annotation/event to the file.<br>
 * This function is optional and can be called only after opening a file in writemode<br>
 * and before closing the file.
 *
 * @param[in] handle
 * File handle.
 *
 * @param[in] onset
 * microseconds since start of recording.
 *
 * @param[in] duration
 * microseconds, > 0 or -1 if not used.
 *
 * @param[in] description
 * A null-terminated Latin1-string containing the text that describes the event.
 *
 * @return
 * 0 on success, otherwise -1.<br>
 */
EDFLIB_API int edfwrite_annotation_latin1_hr(int handle, long long onset, long long duration, const char *description);

#if defined(__GNUC__)
EDFLIB_API int edfwrite_annotation_latin1(int handle, long long onset, long long duration, const char *description) __attribute__ ((deprecated ("use edfwrite_annotation_latin1_hr()")));
#else
EDFLIB_API int edfwrite_annotation_latin1(int handle, long long onset, long long duration, const char *description);
#endif
/* DEPRECATED!! USE edfwrite_annotation_latin1_hr()
 * writes an annotation/event to the file
 * onset is relative to the start of the file
 * onset and duration are in units of 100 microseconds!     resolution is 0.0001 second!
 * for example: 34.071 seconds must be written as 340710
 * if duration is unknown or not applicable: set a negative number (-1)
 * description is a null-terminated Latin1-string containing the text that describes the event
 * This function is optional and can be called only after opening a file in writemode
 * and before closing the file
 */

/**
 * Sets the datarecord duration. The default value is 1 second.<br>
 * ATTENTION: the argument \p duration is expressed in units of 10 microseconds.<br>
 * So, if you want to set the datarecord duration to 0.1 second, you must write a value of 10000.<br>
 * The datarecord duration must be in the range 0.001 to 60 seconds.<br>
 * This function can be used when you want to use a samplerate<br>
 * which is not an integer. For example, if you want to use a samplerate of 0.5 Hz,<br>
 * set the samplefrequency to 5 Hz and the datarecord duration to 10 seconds,<br>
 * or set the samplefrequency to 1 Hz and the datarecord duration to 2 seconds.<br>
 * This function is optional and can be called after opening a<br>
 * file in writemode and before the first sample write action.
 *
 * @param[in] handle
 * File handle.
 *
 * @param[in] duration
 * Datarecord duration expressed in units of 10 microSecond.
 *
 * @return
 * 0 on success, otherwise -1.<br>
 */
EDFLIB_API int edf_set_datarecord_duration(int handle, int duration);

/**
 * Sets the datarecord duration to a very small value.<br>
 * ATTENTION: the argument \p duration is expressed in units of 1 microSecond.<br>
 * The datarecord duration must be in the range 1 to 9999 microseconds.<br>
 * This function can be used when you want to use a very high samplerate.<br>
 * For example, if you want to use a samplerate of 5 GHz,<br>
 * set the samplefrequency to 5000 Hz and the datarecord duration to 1 micro-second.<br>
 * Do not use this function if not necessary.<br>
 * This function was added to accommodate for high speed ADC's e.g. Digital Sampling Oscilloscopes<br>
 * This function is optional and can be called after opening a<br>
 * file in writemode and before the first sample write action.
 *
 * @param[in] handle
 * File handle.
 *
 * @param[in] duration
 * Datarecord duration expressed in units of 10 microSecond.
 *
 * @return
 * 0 on success, otherwise -1.<br>
 */
EDFLIB_API int edf_set_micro_datarecord_duration(int handle, int duration);

/**
 * Sets the number of annotation signals. The default value is 1.<br>
 * This function is optional and can be called only after opening a file in writemode<br>
 * and before the first sample write action.<br>
 * Normally you don't need to change the default value. Only when the number of annotations<br>
 * you expect to write is more than the number of datarecords in the recording, you can use<br>
 * this function to increase the storage space for annotations.<br>
 *
 * @param[in] handle
 * File handle.
 *
 * @param[in] annot_signals
 * Number of annotation signals, must be in the range 1 - 64 inclusive.
 *
 * @return
 * 0 on success, otherwise -1.<br>
 */
EDFLIB_API int edf_set_number_of_annotation_signals(int handle, int annot_signals);

/**
 * Sets the subsecond starttime expressed in units of 100 nanoseconds.<br>
 * Valid range is 0 to 9999999 inclusive. Default is 0.<br>
 * This function is optional and can be called only after opening a file in writemode<br>
 * and before the first sample write action.<br>
 * It is recommended to use a maximum resolution of not more than 100 microseconds.<br>
 * E.g. use 1234000  to set a starttime offset of 0.1234 seconds (instead of for example 1234217).<br>
 * In other words, leave the last 3 digits at zero.<br>
 *
 * @param[in] handle
 * File handle.
 *
 * @param[in] subsecond
 * Subsecond starttime expressed in units of 100 nanoseconds.
 *
 * @return
 * 0 on success, otherwise -1.<br>
 */
EDFLIB_API int edf_set_subsecond_starttime(int handle, int subsecond);

/**
 * Sets the preferred position of the annotation channels(s) before, after or in the middle of the list<br>
 * of regular signals. The default is to put them at the end (after the regular signals).<br>
 * This function is optional and can be called only after opening a file in writemode<br>
 * and before the first sample write action.<br>
 *
 * @param[in] handle
 * File handle.
 *
 * @param[in] pos
 * Preferred position of the annotation channel(s):<br>
 * EDF_ANNOT_IDX_POS_START<br>
 * EDF_ANNOT_IDX_POS_MIDDLE<br>
 * EDF_ANNOT_IDX_POS_END<br>
 *
 * @return
 * 0 on success, otherwise -1.<br>
 */
EDFLIB_API int edf_set_annot_chan_idx_pos(int handle, int pos);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif






