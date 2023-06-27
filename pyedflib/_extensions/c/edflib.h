/*
*****************************************************************************
*
* Copyright (c) 2009 - 2023 Teunis van Beelen
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

/****************************************************************************
 *
 * For more info about the EDF and EDF+ format, visit: http://edfplus.info/specs/
 *
 * For more info about the BDF and BDF+ format, visit: http://www.teuniz.net/edfbrowser/bdfplus%20format%20description.html
 *
 * note: In EDF, the resolution (or sensitivity) (e.g. uV/bit) and offset are stored using four parameters:
 * digital maximum and minimum, and physical maximum and minimum.
 * Here, digital means the raw data coming from a sensor or ADC. Physical means the units like uV.
 * The resolution in units per least significant bit is calculated as follows:
 *
 * units per bit = (physical max - physical min) / (digital max - digital min)
 *
 * The digital offset is calculated as follows:
 *
 * offset = (physical max / units per bit) - digital max
 *
 * For a better explanation about the relation between digital data and physical data,
 * read the document "Coding Schemes Used with Data Converters" (PDF):
 *
 * https://www.ti.com/general/docs/lit/getliterature.tsp?baseLiteratureNumber=sbaa042
 *
 * note: An EDF file usually contains multiple so-called datarecords. One datarecord usually has a duration of one second (this is the default but it is not mandatory!).
 * In that case a file with a duration of five minutes contains 300 datarecords. The duration of a datarecord can be freely chosen but, if possible, use values from
 * 0.1 to 1 second for easier handling. Just make sure that the total size of one datarecord, expressed in bytes, does not exceed 10MByte (15MBytes for BDF(+)).
 *
 * The RECOMMENDATION of a maximum datarecord size of 61440 bytes in the EDF and EDF+ specification was useful in the time people were still using DOS as their main operating system.
 * Using DOS and fast (near) pointers (16-bit pointers), the maximum allocatable block of memory was 64KByte.
 * This is not a concern anymore so the maximum datarecord size now is limited to 10MByte for EDF(+) and 15MByte for BDF(+). This helps to accommodate for higher sampling rates
 * used by modern Analog to Digital Converters.
 *
 * EDF header character encoding: The EDF specification says that only (printable) ASCII characters are allowed.
 * When writing the header info, EDFlib will assume you are using Latin1 encoding and it will automatically convert
 * characters with accents, umlauts, tilde, etc. to their "normal" equivalent without the accent/umlaut/tilde/etc.
 * in order to create a valid EDF file.
 * The description of an EDF+ annotation on the other hand, is always encoded in UTF-8 (which is forward compatible with ASCII).
 *
 * The sample frequency of a signal is calculated as follows: sf = (smp_in_datarecord * EDFLIB_TIME_DIMENSION) / datarecord_duration
 *
 * Annotation signals
 * ==================
 *
 * EDF+ and BDF+ store the annotations in one or more signals (in order to be backwards compatible with EDF and BDF).
 * The numbering of the signals in the file is zero based (starts at 0). Signals used for annotations are skipped by EDFlib.
 * This means that the annotationsignal(s) in the file are hidden.
 * Use the function edf_get_annotation() to get the annotations.
 *
 * So, when a file contains 5 signals and the third signal is an annotations signal, the library will
 * report that there are only 4 signals in the file.
 * The library will "map" the signal numbers as follows: 0->0, 1->1, 2->3, 3->4.
 * This way you don't need to worry about which signals are annotationsignals, the library will take care of it.
 *
 * How the library stores time values
 * ==================================
 *
 * To avoid rounding errors, the library stores some time values in variables of type long long int.
 * In order not to lose the sub-second precision, all time values are scaled with a scaling factor: 10000000.
 * This will limit the time resolution to 100 nanoseconds. To calculate the amount of seconds, divide
 * the timevalue by 10000000 or use the macro EDFLIB_TIME_DIMENSION which is declared in edflib.h.
 * The following variables use this scaling when you open a file in read mode: "file_duration", "starttime_subsecond" and "onset".
 *
 * EDFlib and thread-safety
 * ========================
 * The following functions are always MT-unsafe:
 * edfopen_file_readonly()   (race condition)
 * edfclose_file()           (race condition)
 * edflib_get_handle()       (race condition)
 *
 * When writing to or reading from the same file, all EDFlib functions are MT-unsafe (race condition).
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

/* the following defines are used in the member "filetype" of the edf_hdr_struct
   and as return value for the function edfopen_file_readonly() */
#define EDFLIB_FILETYPE_EDF                  (0)
#define EDFLIB_FILETYPE_EDFPLUS              (1)
#define EDFLIB_FILETYPE_BDF                  (2)
#define EDFLIB_FILETYPE_BDFPLUS              (3)
#define EDFLIB_MALLOC_ERROR                 (-1)
#define EDFLIB_NO_SUCH_FILE_OR_DIRECTORY    (-2)

/* when this error occurs, try to open the file with EDFbrowser,
   it will give you full details about the cause of the error. */
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

typedef struct edf_param_struct
{                                /* this structure contains all the relevant EDF-signal parameters of one signal */
  char   label[17];              /* label (name) of the signal, null-terminated string */
  long long smp_in_file;         /* number of samples of this signal in the file */
  double phys_max;               /* physical maximum, usually the maximum input of the ADC */
  double phys_min;               /* physical minimum, usually the minimum input of the ADC */
  int    dig_max;                /* digital maximum, usually the maximum output of the ADC, cannot not be higher than 32767 for EDF or 8388607 for BDF */
  int    dig_min;                /* digital minimum, usually the minimum output of the ADC, cannot not be lower than -32768 for EDF or -8388608 for BDF */
  int    smp_in_datarecord;      /* number of samples of this signal in a datarecord, if the datarecord has a duration of one second (default), then it equals the sample rate */
  char   physdimension[9];       /* physical dimension (uV, bpm, mA, etc.), null-terminated string */
  char   prefilter[81];          /* null-terminated string */
  char   transducer[81];         /* null-terminated string */
} edflib_param_t;

typedef struct edf_annotation_struct
{                                                       /* this structure is used for annotations */
        long long onset;                                /* onset time of the event, expressed in units of 100 nanoseconds and relative to the start of the file */
        long long duration_l;                           /* duration time, expressed in units of 100 nanoseconds, if less than zero: unused or not applicable */
        char duration[20];                              /* duration time, this is a null-terminated ASCII text-string */
        char annotation[EDFLIB_MAX_ANNOTATION_LEN + 1]; /* description of the event in UTF-8, this is a null terminated string */
} edflib_annotation_t;

typedef struct edf_hdr_struct
{                                 /* this structure contains all the relevant EDF header info and will be filled when calling the function edf_open_file_readonly() */
  int       handle;               /* a handle (identifier) used to distinguish the different files */
  int       filetype;             /* 0: EDF, 1: EDF+, 2: BDF, 3: BDF+, a negative number means an error */
  int       edfsignals;           /* number of EDF signals in the file, annotation channels are NOT included */
  long long file_duration;        /* duration of the file expressed in units of 100 nanoseconds */
  int       startdate_day;
  int       startdate_month;
  int       startdate_year;
  long long starttime_subsecond;  /* starttime offset expressed in units of 100 nanoseconds. Is always less than 10000000 (one second). Only used by EDF+ and BDF+ */
  int       starttime_second;
  int       starttime_minute;
  int       starttime_hour;
  char      patient[81];                                  /* null-terminated string, contains patient field of header, is always empty when filetype is EDFPLUS or BDFPLUS */
  char      recording[81];                                /* null-terminated string, contains recording field of header, is always empty when filetype is EDFPLUS or BDFPLUS */
  char      patientcode[81];                              /* null-terminated string, is always empty when filetype is EDF or BDF */
  char      sex[16];                                      /* null-terminated string, is always empty when filetype is EDF or BDF */
#if defined(__GNUC__)
  char      gender[16] __attribute__ ((deprecated ("use sex")));      /* DEPRECATED!! use "sex" */
#else
  char      gender[16];  /* DEPRECATED!! use "sex" */
#endif
  char      birthdate[16];                                /* null-terminated string, is always empty when filetype is EDF or BDF */
  int       birthdate_day;                                /* 1 - 31 */
  int       birthdate_month;                              /* 1 - 12 */
  int       birthdate_year;
  char      patient_name[81];                             /* null-terminated string, is always empty when filetype is EDF or BDF */
  char      patient_additional[81];                       /* null-terminated string, is always empty when filetype is EDF or BDF */
  char      admincode[81];                                /* null-terminated string, is always empty when filetype is EDF or BDF */
  char      technician[81];                               /* null-terminated string, is always empty when filetype is EDF or BDF */
  char      equipment[81];                                /* null-terminated string, is always empty when filetype is EDF or BDF */
  char      recording_additional[81];                     /* null-terminated string, is always empty when filetype is EDF or BDF */
  long long datarecord_duration;                          /* duration of a datarecord expressed in units of 100 nanoseconds */
  long long datarecords_in_file;                          /* number of datarecords in the file */
  long long annotations_in_file;                          /* number of annotations in the file */
  edflib_param_t signalparam[EDFLIB_MAXSIGNALS];             /* array of structs which contain the relevant signal parameters */
} edflib_hdr_t;

/*****************  the following functions are used to read files **************************/

EDFLIB_API int edfopen_file_readonly(const char *path, edflib_hdr_t *edfhdr, int read_annotations);
/* opens an existing file for reading
 * path is a null-terminated string containing the path to the file
 * hdr is a pointer to an edf_hdr_struct, all fields in this struct will be overwritten
 * the edf_hdr_struct will be filled with all the relevant header- and signalinfo/parameters

 * read_annotations must have one of the following values:
 *   EDFLIB_DO_NOT_READ_ANNOTATIONS      annotations will not be read (this saves time when opening a very large EDF+ or BDF+ file
 *   EDFLIB_READ_ANNOTATIONS             annotations will be read immediately, stops when an annotation has
 *                                       been found which contains the description "Recording ends"
 *   EDFLIB_READ_ALL_ANNOTATIONS         all annotations will be read immediately

 * returns 0 on success, in case of an error it returns -1 and an error code will be set in the member "filetype" of edflib_hdr_t
 * This function is required if you want to read a file
 */

EDFLIB_API int edfread_physical_samples(int handle, int edfsignal, int n, double *buf);
/* reads n samples from edfsignal, starting from the current sample position indicator, into buf (edfsignal starts at 0)
 * the values are converted to their physical values e.g. microVolts, beats per minute, etc.
 * bufsize should be equal to or bigger than sizeof(double[n])
 * the sample position indicator will be increased with the amount of samples read
 * returns the amount of samples read (this can be less than n or zero!)
 * or -1 in case of an error
 */

EDFLIB_API int edfread_digital_samples(int handle, int edfsignal, int n, int *buf);
/* reads n samples from edfsignal, starting from the current sample position indicator, into buf (edfsignal starts at 0)
 * the values are the "raw" digital values
 * bufsize should be equal to or bigger than sizeof(int[n])
 * the sample position indicator will be increased with the amount of samples read
 * returns the amount of samples read (this can be less than n or zero!)
 * or -1 in case of an error
 */

EDFLIB_API long long edfseek(int handle, int edfsignal, long long offset, int whence);
/* The edfseek() function sets the sample position indicator for the edfsignal pointed to by edfsignal.
 * The new position, measured in samples, is obtained by adding offset samples to the position specified by whence.
 * If whence is set to EDFSEEK_SET, EDFSEEK_CUR, or EDFSEEK_END, the offset is relative to the start of the file,
 * the current position indicator, or end-of-file, respectively.
 * Returns the current offset. Otherwise, -1 is returned.
 * note that every signal has it's own independent sample position indicator and edfseek() affects only one of them
 */

EDFLIB_API long long edftell(int handle, int edfsignal);
/* The edftell() function obtains the current value of the sample position indicator for the edfsignal pointed to by edfsignal.
 * Returns the current offset. Otherwise, -1 is returned
 * note that every signal has it's own independent sample position indicator and edftell() affects only one of them
 */

EDFLIB_API void edfrewind(int handle, int edfsignal);
/* The edfrewind() function sets the sample position indicator for the edfsignal pointed to by edfsignal to the beginning of the file.
 * It is equivalent to: (void) edfseek(int handle, int edfsignal, 0LL, EDFSEEK_SET)
 * note that every signal has it's own independent sample position indicator and edfrewind() affects only one of them
 */

EDFLIB_API int edf_get_annotation(int handle, int n, edflib_annotation_t *annot);
/* Fills the edflib_annotation_t structure with the annotation n, returns 0 on success, otherwise -1
 * The string that describes the annotation/event is encoded in UTF-8
 * To obtain the number of annotations in a file, check edf_hdr_struct -> annotations_in_file.
 * returns 0 on success or -1 in case of an error
 */

/*****************  the following functions are used in read and write mode **************************/

EDFLIB_API int edfclose_file(int handle);
/* closes (and in case of writing, finalizes) the file
 * returns -1 in case of an error, 0 on success
 * this function MUST be called when you are finished reading or writing
 * This function is required after reading or writing. Failing to do so will cause
 * unnecessary memory usage and in case of writing it will cause a corrupted and incomplete file
 */

EDFLIB_API int edflib_version(void);
/* Returns the version number of this library, multiplied by hundred. if version is "1.00" than it will return 100
 */

EDFLIB_API int edflib_is_file_used(const char *path);
/* returns 1 if the file is used, either for reading or writing, otherwise returns 0
 */

EDFLIB_API int edflib_get_number_of_open_files(void);
/* returns the number of open files, either for reading or writing
 */

EDFLIB_API int edflib_get_handle(int file_number);
/* returns the handle of an opened file, either for reading or writing
 * file_number is zero based (starts with 0)
 * returns -1 if the file is not opened
 */

/*****************  the following functions are used to write files **************************/

EDFLIB_API int edfopen_file_writeonly(const char *path, int filetype, int number_of_signals);
/* opens an new file for writing. warning, an already existing file with the same name will be silently overwritten without advance warning!
 * path is a null-terminated string containing the path and name of the file
 * filetype must be EDFLIB_FILETYPE_EDFPLUS or EDFLIB_FILETYPE_BDFPLUS
 * returns a handle on success, you need this handle for the other functions
 * in case of an error it returns a negative number corresponding to one of the following values:
 * EDFLIB_MALLOC_ERROR
 * EDFLIB_NO_SUCH_FILE_OR_DIRECTORY
 * EDFLIB_MAXFILES_REACHED
 * EDFLIB_FILE_ALREADY_OPENED
 * EDFLIB_NUMBER_OF_SIGNALS_INVALID
 * EDFLIB_ARCH_ERROR
 * This function is required if you want to write a file (or use edfopen_file_writeonly_with_params())
 */

EDFLIB_API int edfopen_file_writeonly_with_params(const char *path, int filetype, int number_of_signals, int samplefrequency, double phys_max_min, const char *phys_dim);
/* this is a convenience function that can create a new EDF file and initializes the most important parameters.
 * it assumes that all signals are sharing the same parameters (you can still change them though).
 * warning, an already existing file with the same name will be silently overwritten without advance warning!
 * path is a null-terminated string containing the path and name of the file
 * filetype must be EDFLIB_FILETYPE_EDFPLUS or EDFLIB_FILETYPE_BDFPLUS
 * Sets the sample frequency of all signals. (In reality, it sets the number of samples per datarecord which equals the sample frequency only when
 * the datarecords have a duration of 1 second)
 * Sets the physical maximum of all signals to phys_max_min.
 * Sets the physical minimum of all signals to -phys_max_min.
 * Sets the physical dimension (unit) of all signals ("uV", "BPM", "mA", "Degr.", etc.).
 * phys_dim is a pointer to a NULL-terminated ASCII-string containing the physical dimension of the signals
 * returns a handle on success, you need this handle for the other functions
 * in case of an error it returns a negative number corresponding to one of the following values:
 * EDFLIB_MALLOC_ERROR
 * EDFLIB_NO_SUCH_FILE_OR_DIRECTORY
 * EDFLIB_MAXFILES_REACHED
 * EDFLIB_FILE_ALREADY_OPENED
 * EDFLIB_NUMBER_OF_SIGNALS_INVALID
 * EDFLIB_ARCH_ERROR
 * This function is required if you want to write a file (or use edfopen_file_writeonly())
 */

EDFLIB_API int edf_set_samplefrequency(int handle, int edfsignal, int samplefrequency);
/* Sets the sample frequency of signal edfsignal. In reality, it sets the number of samples in a datarecord
 * which equals the sample frequency only when the datarecords have a duration of 1 second.
 * The effective sample frequency is: samplefrequency / datarecord duration
 * Returns 0 on success, otherwise -1
 * This function is required for every signal and can be called only after opening a
 * file in write mode and before the first sample write action
 */

EDFLIB_API int edf_set_physical_maximum(int handle, int edfsignal, double phys_max);
/* Sets the maximum physical value of signal edfsignal. (the value of the input of the ADC when the output equals the value of "digital maximum")
 * It is the highest value that the equipment is able to record. It does not necessarily mean the signal recorded reaches this level
 * Must be un-equal to physical minimum
 * Returns 0 on success, otherwise -1
 * This function is required for every signal and can be called only after opening a
 * file in write mode and before the first sample write action
 */

EDFLIB_API int edf_set_physical_minimum(int handle, int edfsignal, double phys_min);
/* Sets the minimum physical value of signal edfsignal. (the value of the input of the ADC when the output equals the value of "digital minimum")
 * It is the lowest value that the equipment is able to record. It does not necessarily mean the signal recorded reaches this level
 * Usually this will be (-(phys_max))
 * Must be un-equal to physical maximum
 * Returns 0 on success, otherwise -1
 * This function is required for every signal and can be called only after opening a
 * file in write mode and before the first sample write action
 */

EDFLIB_API int edf_set_digital_maximum(int handle, int edfsignal, int dig_max);
/* Sets the maximum digital value of signal edfsignal. The maximum value is 32767 for EDF+ and 8388607 for BDF+
 * It is the highest value that the equipment is able to record. It does not necessarily mean the signal recorded reaches this level
 * Usually it's the extreme output of the ADC
 * Must be higher than digital minimum
 * Returns 0 on success, otherwise -1
 * This function is required for every signal and can be called only after opening a file in write mode
 * and before the first sample write action
 */

EDFLIB_API int edf_set_digital_minimum(int handle, int edfsignal, int dig_min);
/* Sets the minimum digital value of signal edfsignal. The minimum value is -32768 for EDF+ and -8388608 for BDF+
 * It is the lowest value that the equipment is able to record. It does not necessarily mean the signal recorded reaches this level
 * Usually it's the extreme output of the ADC
 * Usually this will be (-(dig_max + 1))
 * Must be lower than digital maximum
 * Returns 0 on success, otherwise -1
 * This function is required for every signal and can be called only after opening a file in write mode
 * and before the first sample write action
 */

EDFLIB_API int edf_set_label(int handle, int edfsignal, const char *label);
/* Sets the label (name) of signal edfsignal. ("FP1", "SaO2", etc.)
 * label is a pointer to a NULL-terminated ASCII-string containing the label (name) of the signal edfsignal
 * Returns 0 on success, otherwise -1
 * This function is recommended for every signal when you want to write a file
 * and can be called only after opening a file in write mode and before the first sample write action
 */

EDFLIB_API int edf_set_prefilter(int handle, int edfsignal, const char *prefilter);
/* Sets the prefilter of signal edfsignal ("HP:0.1Hz", "LP:75Hz N:50Hz", etc.).
 * prefilter is a pointer to a NULL-terminated ASCII-string containing the prefilter text of the signal edfsignal
 * Returns 0 on success, otherwise -1
 * This function is optional and can be called only after opening a file in write mode and before
 * the first sample write action
 */

EDFLIB_API int edf_set_transducer(int handle, int edfsignal, const char *transducer);
/* Sets the transducer of signal edfsignal ("AgAgCl cup electrodes", etc.).
 * transducer is a pointer to a NULL-terminated ASCII-string containing the transducer text of the signal edfsignal
 * Returns 0 on success, otherwise -1
 * This function is optional and can be called only after opening a file in writemode and before
 * the first sample write action
 */

EDFLIB_API int edf_set_physical_dimension(int handle, int edfsignal, const char *phys_dim);
/* Sets the physical dimension (unit) of signal edfsignal. ("uV", "BPM", "mA", "Degr.", etc.)
 * phys_dim is a pointer to a NULL-terminated ASCII-string containing the physical dimension of the signal edfsignal
 * Returns 0 on success, otherwise -1
 * This function is recommended for every signal when you want to write a file
 * and can be called only after opening a file in write mode and before the first sample write action
 */

EDFLIB_API int edf_set_startdatetime(int handle, int startdate_year, int startdate_month, int startdate_day,
                                     int starttime_hour, int starttime_minute, int starttime_second);
/* Sets the startdate and starttime.
 * year: 1985 - 2084, month: 1 - 12, day: 1 - 31
 * hour: 0 - 23, minute: 0 - 59, second: 0 - 59
 * If not called, the library will use the system date and time at runtime
 * Returns 0 on success, otherwise -1
 * This function is optional and can be called only after opening a file in write mode
 * and before the first sample write action
 * Note: for anonymization purposes, the consensus is to use 1985-01-01 00:00:00 for the startdate and starttime.
 */

EDFLIB_API int edf_set_patientname(int handle, const char *patientname);
/* Sets the patientname. patientname is a pointer to a null-terminated ASCII-string.
 * Returns 0 on success, otherwise -1
 * This function is optional and can be called only after opening a file in writemode
 * and before the first sample write action
 */

EDFLIB_API int edf_set_patientcode(int handle, const char *patientcode);
/* Sets the patientcode. patientcode is a pointer to a null-terminated ASCII-string.
 * Returns 0 on success, otherwise -1
 * This function is optional and can be called only after opening a file in writemode
 * and before the first sample write action
 */

EDFLIB_API int edf_set_sex(int handle, int sex);
/* Sets the sex. 1 is male, 0 is female.
 * Returns 0 on success, otherwise -1
 * This function is optional and can be called only after opening a file in writemode
 * and before the first sample write action
 */

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

EDFLIB_API int edf_set_birthdate(int handle, int birthdate_year, int birthdate_month, int birthdate_day);
/* Sets the birthdate.
 * year: 1800 - 3000, month: 1 - 12, day: 1 - 31
 * This function is optional
 * Returns 0 on success, otherwise -1
 * This function is optional and can be called only after opening a file in writemode
 * and before the first sample write action
 */

EDFLIB_API int edf_set_patient_additional(int handle, const char *patient_additional);
/* Sets the additional patientinfo. patient_additional is a pointer to a null-terminated ASCII-string.
 * Returns 0 on success, otherwise -1
 * This function is optional and can be called only after opening a file in writemode
 * and before the first sample write action
 */

EDFLIB_API int edf_set_admincode(int handle, const char *admincode);
/* Sets the admincode. admincode is a pointer to a null-terminated ASCII-string.
 * Returns 0 on success, otherwise -1
 * This function is optional and can be called only after opening a file in writemode
 * and before the first sample write action
 */

EDFLIB_API int edf_set_technician(int handle, const char *technician);
/* Sets the technicians name. technician is a pointer to a null-terminated ASCII-string.
 * Returns 0 on success, otherwise -1
 * This function is optional and can be called only after opening a file in writemode
 * and before the first sample write action
 */

EDFLIB_API int edf_set_equipment(int handle, const char *equipment);
/* Sets the name of the equipment used during the acquisition. equipment is a pointer to a null-terminated ASCII-string.
 * Returns 0 on success, otherwise -1
 * This function is optional and can be called only after opening a file in writemode
 * and before the first sample write action
 */

EDFLIB_API int edf_set_recording_additional(int handle, const char *recording_additional);
/* Sets the additional recordinginfo. recording_additional is a pointer to a null-terminated ASCII-string.
 * Returns 0 on success, otherwise -1
 * This function is optional and can be called only after opening a file in writemode
 * and before the first sample write action
 */

EDFLIB_API int edfwrite_physical_samples(int handle, double *buf);
/* Writes n physical samples (uV, mA, Ohm) from *buf belonging to one signal
 * where n is the samplefrequency of that signal.
 * The physical samples will be converted to digital samples using the
 * values of physical maximum, physical minimum, digital maximum and digital minimum
 * The number of samples written is equal to the samplefrequency of the signal
 * Size of buf should be equal to or bigger than sizeof(double[samplefrequency])
 * Call this function for every signal in the file. The order is important!
 * When there are 4 signals in the file,  the order of calling this function
 * must be: signal 0, signal 1, signal 2, signal 3, signal 0, signal 1, signal 2, etc.
 * Returns 0 on success, otherwise -1
 */

EDFLIB_API int edf_blockwrite_physical_samples(int handle, double *buf);
/* Writes physical samples (uV, mA, Ohm) from *buf
 * buf must be filled with samples from all signals, starting with n samples of signal 0, n samples of signal 1, n samples of signal 2, etc.
 * where n is the samplefrequency of that signal.
 * buf must be filled with samples from all signals, starting with signal 0, 1, 2, etc.
 * one block equals one second
 * The physical samples will be converted to digital samples using the
 * values of physical maximum, physical minimum, digital maximum and digital minimum
 * The number of samples written is equal to the sum of the samplefrequencies of all signals
 * Size of buf should be equal to or bigger than sizeof(double) multiplied by the sum of the samplefrequencies of all signals
 * Returns 0 on success, otherwise -1
 */

EDFLIB_API int edfwrite_digital_short_samples(int handle, short *buf);
/* Writes n "raw" digital samples from *buf belonging to one signal
 * where n is the samplefrequency of that signal.
 * The samples will be written to the file without any conversion.
 * Because the size of a short is 16-bit, do not use this function with BDF (24-bit)
 * The number of samples written is equal to the samplefrequency of the signal
 * Size of buf should be equal to or bigger than sizeof(short[samplefrequency])
 * Call this function for every signal in the file. The order is important!
 * When there are 4 signals in the file,  the order of calling this function
 * must be: signal 0, signal 1, signal 2, signal 3, signal 0, signal 1, signal 2, etc.
 * Returns 0 on success, otherwise -1
 */

EDFLIB_API int edfwrite_digital_samples(int handle, int *buf);
/* Writes n "raw" digital samples from *buf belonging to one signal
 * where n is the samplefrequency of that signal.
 * The 16 (or 24 in case of BDF) least significant bits of the sample will be written to the
 * file without any conversion.
 * The number of samples written is equal to the samplefrequency of the signal
 * Size of buf should be equal to or bigger than sizeof(int[samplefrequency])
 * Call this function for every signal in the file. The order is important!
 * When there are 4 signals in the file,  the order of calling this function
 * must be: signal 0, signal 1, signal 2, signal 3, signal 0, signal 1, signal 2, etc.
 * Returns 0 on success, otherwise -1
 */

EDFLIB_API int edf_blockwrite_digital_3byte_samples(int handle, void *buf);
/* Writes "raw" digital samples from *buf.
 * buf must be filled with samples from all signals, starting with n samples of signal 0, n samples of signal 1, n samples of signal 2, etc.
 * where n is the samplefrequency of that signal.
 * One block equals one second. One sample equals 3 bytes, order is little endian (least significant byte first)
 * Encoding is second's complement, most significant bit of most significant byte is the sign-bit
 * The samples will be written to the file without any conversion.
 * Because the size of a 3-byte sample is 24-bit, this function can only be used when writing a BDF file
 * The number of samples written is equal to the sum of the samplefrequencies of all signals.
 * Size of buf should be equal to or bigger than: the sum of the samplefrequencies of all signals x 3 bytes
 * Returns 0 on success, otherwise -1
 */

EDFLIB_API int edf_blockwrite_digital_short_samples(int handle, short *buf);
/* Writes "raw" digital samples from *buf.
 * buf must be filled with samples from all signals, starting with n samples of signal 0, n samples of signal 1, n samples of signal 2, etc.
 * where n is the samplefrequency of that signal.
 * One block equals one second.
 * The samples will be written to the file without any conversion.
 * Because the size of a short is 16-bit, do not use this function with BDF (24-bit)
 * The number of samples written is equal to the sum of the samplefrequencies of all signals.
 * Size of buf should be equal to or bigger than sizeof(short) multiplied by the sum of the samplefrequencies of all signals
 * Returns 0 on success, otherwise -1
 */

EDFLIB_API int edf_blockwrite_digital_samples(int handle, int *buf);
/* Writes "raw" digital samples from *buf.
 * buf must be filled with samples from all signals, starting with n samples of signal 0, n samples of signal 1, n samples of signal 2, etc.
 * where n is the samplefrequency of that signal.
 * One block equals one second.
 * The 16 (or 24 in case of BDF) least significant bits of the sample will be written to the
 * file without any conversion.
 * The number of samples written is equal to the sum of the samplefrequencies of all signals.
 * Size of buf should be equal to or bigger than sizeof(int) multiplied by the sum of the samplefrequencies of all signals
 * Returns 0 on success, otherwise -1
 */

EDFLIB_API int edfwrite_annotation_utf8_hr(int handle, long long onset, long long duration, const char *description);
/* writes an annotation/event to the file
 * onset is relative to the start of the file
 * onset and duration are in units of 1 microSecond     resolution is 0.000001 second
 * for example: 34.071 seconds must be written as 34071000
 * if duration is unknown or not applicable: set a negative number (-1)
 * description is a null-terminated UTF8-string containing the text that describes the event
 * This function is optional and can be called only after opening a file in writemode
 * and before closing the file
 */

#if defined(__GNUC__)
EDFLIB_API int edfwrite_annotation_utf8(int handle, long long onset, long long duration, const char *description) __attribute__ ((deprecated ("use edfwrite_annotation_utf8_hr()")));
#else
EDFLIB_API int edfwrite_annotation_utf8(int handle, long long onset, long long duration, const char *description);
#endif
/* DEPRECATED!! USE edfwrite_annotation_utf8_hr()
 * writes an annotation/event to the file
 * onset is relative to the start of the file
 * onset and duration are in units of 100 microSeconds!     resolution is 0.0001 second!
 * for example: 34.071 seconds must be written as 340710
 * if duration is unknown or not applicable: set a negative number (-1)
 * description is a null-terminated UTF8-string containing the text that describes the event
 * This function is optional and can be called only after opening a file in writemode
 * and before closing the file
 */

EDFLIB_API int edfwrite_annotation_latin1_hr(int handle, long long onset, long long duration, const char *description);
/* writes an annotation/event to the file
 * onset is relative to the start of the file
 * onset and duration are in units of 1 microSecond     resolution is 0.000001 second
 * for example: 34.071 seconds must be written as 34071000
 * if duration is unknown or not applicable: set a negative number (-1)
 * description is a null-terminated Latin1-string containing the text that describes the event
 * This function is optional and can be called only after opening a file in writemode
 * and before closing the file
 */

#if defined(__GNUC__)
EDFLIB_API int edfwrite_annotation_latin1(int handle, long long onset, long long duration, const char *description) __attribute__ ((deprecated ("use edfwrite_annotation_latin1_hr()")));
#else
EDFLIB_API int edfwrite_annotation_latin1(int handle, long long onset, long long duration, const char *description);
#endif
/* DEPRECATED!! USE edfwrite_annotation_latin1_hr()
 * writes an annotation/event to the file
 * onset is relative to the start of the file
 * onset and duration are in units of 100 microSeconds!     resolution is 0.0001 second!
 * for example: 34.071 seconds must be written as 340710
 * if duration is unknown or not applicable: set a negative number (-1)
 * description is a null-terminated Latin1-string containing the text that describes the event
 * This function is optional and can be called only after opening a file in writemode
 * and before closing the file
 */

EDFLIB_API int edf_set_datarecord_duration(int handle, int duration);
/* Sets the datarecord duration. The default value is 1 second.
 * ATTENTION: the argument "duration" is expressed in units of 10 microSeconds!
 * So, if you want to set the datarecord duration to 0.1 second, you must give
 * the argument "duration" a value of "10000".
 * This function is optional, normally you don't need to change the default value.
 * The datarecord duration must be in the range 0.001 to 60 seconds.
 * Returns 0 on success, otherwise -1
 * This function is NOT REQUIRED but can be called after opening a
 * file in writemode and before the first sample write action.
 * This function can be used when you want to use a samplerate
 * which is not an integer. For example, if you want to use a samplerate of 0.5 Hz,
 * set the samplefrequency to 5 Hz and the datarecord duration to 10 seconds,
 * or set the samplefrequency to 1 Hz and the datarecord duration to 2 seconds.
 * Do not use this function if not necessary.
 */

EDFLIB_API int edf_set_micro_datarecord_duration(int handle, int duration);
/* Sets the datarecord duration to a very small value.
 * ATTENTION: the argument "duration" is expressed in units of 1 microSecond!
 * This function is optional, normally you don't need to change the default value.
 * The datarecord duration must be in the range 1 to 9999 micro-seconds.
 * Returns 0 on success, otherwise -1
 * This function is NOT REQUIRED but can be called after opening a
 * file in writemode and before the first sample write action.
 * This function can be used when you want to use a very high samplerate.
 * For example, if you want to use a samplerate of 5 GHz,
 * set the samplefrequency to 5000 Hz and the datarecord duration to 1 micro-second.
 * Do not use this function if not necessary.
 * This function was added to accommodate for high speed ADC's e.g. Digital Sampling Oscilloscopes
 */

EDFLIB_API int edf_set_number_of_annotation_signals(int handle, int annot_signals);
/* Sets the number of annotation signals. The default value is 1
 * This function is optional and can be called only after opening a file in writemode
 * and before the first sample write action
 * Normally you don't need to change the default value. Only when the number of annotations
 * you want to write is higher than the number of datarecords in the recording, you can use
 * this function to increase the storage space for annotations
 * Minimum is 1, maximum is 64
 * Returns 0 on success, otherwise -1
 */

EDFLIB_API int edf_set_subsecond_starttime(int handle, int subsecond);
/* Sets the subsecond starttime expressed in units of 100 nanoseconds
 * Valid range is 0 to 9999999 inclusive. Default is 0
 * This function is optional and can be called only after opening a file in writemode
 * and before the first sample write action
 * Returns 0 on success, otherwise -1
 * It is strongly recommended to use a maximum resolution of no more than 100 micro-Seconds.
 * e.g. use 1234000  to set a starttime offset of 0.1234 seconds (instead of 1234567)
 * in other words, leave the last 3 digits at zero
 */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif






