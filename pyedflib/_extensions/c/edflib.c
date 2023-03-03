/*
*****************************************************************************
*
* Copyright (c) 2009 - 2020 Teunis van Beelen
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




/* compile with options "-D_LARGEFILE64_SOURCE -D_LARGEFILE_SOURCE" */

#pragma warning( disable : 4996 ) // ignore unsafe strncpy
#pragma warning( disable : 4244 ) // ignore precision loss

#include "edflib.h"


#define EDFLIB_VERSION  (117)
#define EDFLIB_MAXFILES  (64)


#if defined(__APPLE__) || defined(__MACH__) || defined(__APPLE_CC__) || defined(__HAIKU__)

#define fopeno fopen

#else

#define fseeko fseeko64
#define ftello ftello64
#define fopeno fopen64

#endif


#ifdef _WIN32

#ifndef __MINGW32__
/* needed for visual c */
#undef fseeko
#define fseeko _fseeki64

#undef ftello
#define ftello _ftelli64

#undef fopeno
#define fopeno fopen

#endif

#endif



/* max size of annotationtext */
#define EDFLIB_WRITE_MAX_ANNOTATION_LEN  (40)

/* bytes in datarecord for EDF annotations, must be an integer multiple of three and two */
#define EDFLIB_ANNOTATION_BYTES  (114)

/* for writing only */
#define EDFLIB_MAX_ANNOTATION_CHANNELS  (64)

#define EDFLIB_ANNOT_MEMBLOCKSZ  (1000)


struct edfparamblock{
        char   label[17];
        char   transducer[81];
        char   physdimension[9];
        double phys_min;
        double phys_max;
        int    dig_min;
        int    dig_max;
        char   prefilter[81];
        int    smp_per_record;
        char   reserved[33];
        double offset;
        int    buf_offset;
        double bitvalue;
        int    annotation;
        long long sample_pntr;
      };

struct edfhdrblock{
        FILE      *file_hdl;
        char      path[1024];
        int       writemode;
        char      version[32];
        char      patient[81];
        char      recording[81];
        char      plus_patientcode[81];
        char      plus_gender[16];
        char      plus_birthdate[16];
        char      plus_patient_name[81];
        char      plus_patient_additional[81];
        char      plus_startdate[16];
        char      plus_admincode[81];
        char      plus_technician[81];
        char      plus_equipment[81];
        char      plus_recording_additional[81];
        long long l_starttime;
        int       startdate_day;
        int       startdate_month;
        int       startdate_year;
        int       starttime_second;
        int       starttime_minute;
        int       starttime_hour;
        char      reserved[45];
        int       hdrsize;
        int       edfsignals;
        long long datarecords;
        int       recordsize;
        int       annot_ch[EDFLIB_MAXSIGNALS];
        int       nr_annot_chns;
        int       mapped_signals[EDFLIB_MAXSIGNALS];
        int       edf;
        int       edfplus;
        int       bdf;
        int       bdfplus;
        int       discontinuous;
        int       signal_write_sequence_pos;
        long long starttime_offset;
        double    data_record_duration;
        long long long_data_record_duration;
        int       annots_in_file;
        int       annotlist_sz;
        int       total_annot_bytes;
        int       eq_sf;
        char      *wrbuf;
        int       wrbufsize;
        struct edfparamblock *edfparam;
      };


static struct edf_annotationblock{
        long long onset;
        char duration[16];
        char annotation[EDFLIB_MAX_ANNOTATION_LEN + 1];
       } *annotationslist[EDFLIB_MAXFILES];


static struct edf_write_annotationblock{
        long long onset;
        long long duration;
        char annotation[EDFLIB_WRITE_MAX_ANNOTATION_LEN + 1];
       } *write_annotationslist[EDFLIB_MAXFILES];

static int edf_files_open=0;

static struct edfhdrblock *hdrlist[EDFLIB_MAXFILES];


static struct edfhdrblock * edflib_check_edf_file(FILE *, int *, int);
static int edflib_repair_file_size(const char *path, struct edfhdrblock *edfhdr);
static int edflib_is_integer_number(char *);
static int edflib_is_number(char *);
static long long edflib_get_long_duration(char *);
static int edflib_get_annotations(struct edfhdrblock *, int, int);
static int edflib_is_duration_number(char *);
static int edflib_is_onset_number(char *);
static long long edflib_get_long_time(char *);
static int edflib_write_edf_header(struct edfhdrblock *);
static void edflib_latin1_to_ascii(char *, int);
static void edflib_latin12utf8(char *, int);
static void edflib_remove_padding_trailing_spaces(char *);
static int edflib_atoi_nonlocalized(const char *);
static double edflib_atof_nonlocalized(const char *);
static int edflib_snprint_number_nonlocalized(char *, double, int);
/*
static int edflib_sprint_int_number_nonlocalized(char *, int, int, int);
*/
static int edflib_snprint_ll_number_nonlocalized(char *, long long, int, int, int);
static int edflib_fprint_int_number_nonlocalized(FILE *, int, int, int);
static int edflib_fprint_ll_number_nonlocalized(FILE *, long long, int, int);
static int edflib_write_tal(struct edfhdrblock *, FILE *);
static int edflib_strlcpy(char *, const char *, int);
static int edflib_strlcat(char *, const char *, int);




int edflib_is_file_used(const char *path)
{
  int i, file_used=0;

  for(i=0; i<EDFLIB_MAXFILES; i++)
  {
    if(hdrlist[i]!=NULL)
    {
      if(!(strcmp(path, hdrlist[i]->path)))
      {
		file_used= 1;

        break;
      }
    }
  }

  return file_used ;
}


int edflib_get_number_of_open_files()
{
  return edf_files_open;
}


int edflib_get_handle(int file_number)
{
  int i, file_count=0;

  for(i=0; i<EDFLIB_MAXFILES; i++)
  {
    if(hdrlist[i]!=NULL)
    {
      if(file_count++ == file_number)
      {
        return i;
      }
    }
  }

  return -1;
}


int edfopen_file_readonly(const char *path, struct edf_hdr_struct *edfhdr, int read_annotations, int check_file_size)
{
  int i, j,
      channel,
      edf_error;

  FILE *file;

  struct edfhdrblock *hdr;


  if(read_annotations<0)
  {
    edfhdr->filetype = EDFLIB_INVALID_READ_ANNOTS_VALUE;

    return -1;
  }

  if(read_annotations>2)
  {
    edfhdr->filetype = EDFLIB_INVALID_READ_ANNOTS_VALUE;

    return -1;
  }

  if(check_file_size<0)	
  {	
    edfhdr->filetype = EDFLIB_INVALID_CHECK_SIZE_VALUE;	


    return -1;	
  }	

  if(check_file_size>2)	
  {	
    edfhdr->filetype = EDFLIB_INVALID_CHECK_SIZE_VALUE;	

    return -1;
  }
  
  memset(edfhdr, 0, sizeof(struct edf_hdr_struct));

  if(edf_files_open>=EDFLIB_MAXFILES)
  {
    edfhdr->filetype = EDFLIB_MAXFILES_REACHED;

    return -1;
  }

  for(i=0; i<EDFLIB_MAXFILES; i++)
  {
    if(hdrlist[i]!=NULL)
    {
      if(!(strcmp(path, hdrlist[i]->path)))
      {
        edfhdr->filetype = EDFLIB_FILE_ALREADY_OPENED;

        return -1;
      }
    }
  }

  file = fopeno(path, "rb");
  if(file==NULL)
  {
    edfhdr->filetype = EDFLIB_NO_SUCH_FILE_OR_DIRECTORY;

    return -1;
  }

  hdr = edflib_check_edf_file(file, &edf_error,check_file_size);
  if (hdr==NULL && edf_error == EDFLIB_FILE_CONTAINS_FORMAT_ERRORS && check_file_size == EDFLIB_REPAIR_FILE_SIZE_IF_WRONG)
  {
    hdr = edflib_check_edf_file(file, &edf_error, EDFLIB_DO_NOT_CHECK_FILE_SIZE);
    fclose(file);
    edflib_repair_file_size(path, hdr);
    free(hdr->edfparam);
    free(hdr);		
    file = fopeno(path, "rb");
    hdr = edflib_check_edf_file(file, &edf_error, EDFLIB_CHECK_FILE_SIZE);
  }

  if(hdr==NULL)
  {
    edfhdr->filetype = edf_error;

    fclose(file);

    return -1;
  }

  if(hdr->discontinuous)
  {
    edfhdr->filetype = EDFLIB_FILE_IS_DISCONTINUOUS;

    free(hdr->edfparam);
    free(hdr);

    fclose(file);

    return -1;
  }

  hdr->writemode = 0;

  for(i=0; i<EDFLIB_MAXFILES; i++)
  {
    if(hdrlist[i]==NULL)
    {
      hdrlist[i] = hdr;

      edfhdr->handle = i;

      break;
    }
  }

  if((hdr->edf)&&(!(hdr->edfplus)))
  {
    edfhdr->filetype = EDFLIB_FILETYPE_EDF;
  }

  if(hdr->edfplus)
  {
    edfhdr->filetype = EDFLIB_FILETYPE_EDFPLUS;
  }

  if((hdr->bdf)&&(!(hdr->bdfplus)))
  {
    edfhdr->filetype = EDFLIB_FILETYPE_BDF;
  }

  if(hdr->bdfplus)
  {
    edfhdr->filetype = EDFLIB_FILETYPE_BDFPLUS;
  }

  edfhdr->edfsignals = hdr->edfsignals - hdr->nr_annot_chns;
  edfhdr->file_duration = hdr->long_data_record_duration * hdr->datarecords;
  edfhdr->startdate_day = hdr->startdate_day;
  edfhdr->startdate_month = hdr->startdate_month;
  edfhdr->startdate_year = hdr->startdate_year;
  edfhdr->starttime_hour = hdr->starttime_hour;
  edfhdr->starttime_second = hdr->starttime_second;
  edfhdr->starttime_minute = hdr->starttime_minute;
  edfhdr->starttime_subsecond = hdr->starttime_offset;
  edfhdr->datarecords_in_file = hdr->datarecords;
  edfhdr->datarecord_duration = hdr->long_data_record_duration;

  annotationslist[edfhdr->handle] = NULL;

  hdr->annotlist_sz = 0;

  hdr->annots_in_file = 0;

  edfhdr->annotations_in_file = 0LL;

  if((!(hdr->edfplus))&&(!(hdr->bdfplus)))
  {
    edflib_strlcpy(edfhdr->patient, hdr->patient, 81);
    edflib_strlcpy(edfhdr->recording, hdr->recording, 81);
    edfhdr->patientcode[0] = 0;
    edfhdr->gender[0] = 0;
    edfhdr->birthdate[0] = 0;
    edfhdr->patient_name[0] = 0;
    edfhdr->patient_additional[0] = 0;
    edfhdr->admincode[0] = 0;
    edfhdr->technician[0] = 0;
    edfhdr->equipment[0] = 0;
    edfhdr->recording_additional[0] = 0;
  }
  else
  {
    edfhdr->patient[0] = 0;
    edfhdr->recording[0] = 0;
    edflib_strlcpy(edfhdr->patientcode, hdr->plus_patientcode, 81);
    edflib_strlcpy(edfhdr->gender, hdr->plus_gender, 16);
    edflib_strlcpy(edfhdr->birthdate, hdr->plus_birthdate, 16);
    edflib_strlcpy(edfhdr->patient_name, hdr->plus_patient_name, 81);
    edflib_strlcpy(edfhdr->patient_additional, hdr->plus_patient_additional, 81);
    edflib_strlcpy(edfhdr->admincode, hdr->plus_admincode, 81);
    edflib_strlcpy(edfhdr->technician, hdr->plus_technician, 81);
    edflib_strlcpy(edfhdr->equipment, hdr->plus_equipment, 81);
    edflib_strlcpy(edfhdr->recording_additional, hdr->plus_recording_additional, 81);

    if((read_annotations==EDFLIB_READ_ANNOTATIONS)||(read_annotations==EDFLIB_READ_ALL_ANNOTATIONS))
    {
      if(edflib_get_annotations(hdr, edfhdr->handle, read_annotations))
      {
        edfhdr->filetype = EDFLIB_FILE_CONTAINS_FORMAT_ERRORS;

        fclose(file);

        free(hdr->edfparam);
        hdr->edfparam = NULL;
        free(hdr);
        hdr = NULL;
        hdrlist[edfhdr->handle] = NULL;
        free(annotationslist[edfhdr->handle]);
        annotationslist[edfhdr->handle] = NULL;

        return -1;
      }

      edfhdr->starttime_subsecond = hdr->starttime_offset;
    }

    edfhdr->annotations_in_file = hdr->annots_in_file;
  }

  edflib_strlcpy(hdr->path, path, 1024);

  edf_files_open++;

  j = 0;

  for(i=0; i<hdr->edfsignals; i++)
  {
    if(!(hdr->edfparam[i].annotation))
    {
      hdr->mapped_signals[j++] = i;
    }
  }

  for(i=0; i<edfhdr->edfsignals; i++)
  {
    channel = hdr->mapped_signals[i];

    edflib_strlcpy(edfhdr->signalparam[i].label, hdr->edfparam[channel].label, 17);
    edflib_strlcpy(edfhdr->signalparam[i].transducer, hdr->edfparam[channel].transducer, 81);
    edflib_strlcpy(edfhdr->signalparam[i].physdimension, hdr->edfparam[channel].physdimension, 9);
    edflib_strlcpy(edfhdr->signalparam[i].prefilter, hdr->edfparam[channel].prefilter, 81);
    edfhdr->signalparam[i].smp_in_file = hdr->edfparam[channel].smp_per_record * hdr->datarecords;
    edfhdr->signalparam[i].phys_max = hdr->edfparam[channel].phys_max;
    edfhdr->signalparam[i].phys_min = hdr->edfparam[channel].phys_min;
    edfhdr->signalparam[i].dig_max = hdr->edfparam[channel].dig_max;
    edfhdr->signalparam[i].dig_min = hdr->edfparam[channel].dig_min;
    edfhdr->signalparam[i].smp_in_datarecord = hdr->edfparam[channel].smp_per_record;
  }

  return 0;
}


int edfclose_file(int handle)
{
  struct edf_write_annotationblock *annot2;

  int i, j, k, n, p, err,
      datrecsize,
      nmemb;

  long long offset,
            datarecords;

  char str[EDFLIB_ANNOTATION_BYTES * 2];

  struct edfhdrblock *hdr;


  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  hdr = hdrlist[handle];

  if(hdr->writemode)
  {
    if(hdr->datarecords == 0LL)
    {
      err = edflib_write_edf_header(hdr);
      if(err)
      {
        fclose(hdr->file_hdl);

        free(hdr->edfparam);

        free(hdr->wrbuf);

        free(hdr);

        hdrlist[handle] = NULL;

        free(write_annotationslist[handle]);

        write_annotationslist[handle] = NULL;

        edf_files_open--;

        return err;

      }

      for(k=0; k<hdr->annots_in_file; k++)
      {
        annot2 = write_annotationslist[handle] + k;

        p = edflib_fprint_ll_number_nonlocalized(hdr->file_hdl, (hdr->datarecords * hdr->long_data_record_duration + hdr->starttime_offset) / EDFLIB_TIME_DIMENSION, 0, 1);

        if((hdr->long_data_record_duration % EDFLIB_TIME_DIMENSION) || (hdr->starttime_offset))
        {
          fputc('.', hdr->file_hdl);
          p++;
          p += edflib_fprint_ll_number_nonlocalized(hdr->file_hdl, (hdr->datarecords * hdr->long_data_record_duration + hdr->starttime_offset) % EDFLIB_TIME_DIMENSION, 7, 0);
        }
        fputc(20, hdr->file_hdl);
        fputc(20, hdr->file_hdl);
        p += 2;
        for(; p<hdr->total_annot_bytes; p++)
        {
          fputc(0, hdr->file_hdl);
        }

        hdr->datarecords++;
      }
    }

    if(hdr->datarecords<100000000LL)
    {
      fseeko(hdr->file_hdl, 236LL, SEEK_SET);
      p = edflib_fprint_int_number_nonlocalized(hdr->file_hdl, (int)(hdr->datarecords), 0, 0);
      if(p < 2)
      {
        fputc(' ', hdr->file_hdl);
      }
    }

    datarecords = 0LL;

    offset = (long long)((hdr->edfsignals + hdr->nr_annot_chns + 1) * 256);

    datrecsize = hdr->total_annot_bytes;

    for(i=0; i<hdr->edfsignals; i++)
    {
      if(hdr->edf)
      {
        offset += (long long)(hdr->edfparam[i].smp_per_record * 2);

        datrecsize += (hdr->edfparam[i].smp_per_record * 2);
      }
      else
      {
        offset += (long long)(hdr->edfparam[i].smp_per_record * 3);

        datrecsize += (hdr->edfparam[i].smp_per_record * 3);
      }
    }

    j = 0;

    for(k=0; k<hdr->annots_in_file; k++)
    {
      annot2 = write_annotationslist[handle] + k;

      annot2->onset += hdr->starttime_offset / 1000LL;

      p = 0;

      if(j==0)  // first annotation signal
      {
        if(fseeko(hdr->file_hdl, offset, SEEK_SET))
        {
          break;
        }

        p += edflib_snprint_ll_number_nonlocalized(str, (datarecords * hdr->long_data_record_duration + hdr->starttime_offset) / EDFLIB_TIME_DIMENSION, 0, 1, EDFLIB_ANNOTATION_BYTES * 2);

        if((hdr->long_data_record_duration % EDFLIB_TIME_DIMENSION) || (hdr->starttime_offset))
        {
          str[p++] = '.';
          n = edflib_snprint_ll_number_nonlocalized(str + p, (datarecords * hdr->long_data_record_duration + hdr->starttime_offset) % EDFLIB_TIME_DIMENSION, 7, 0, (EDFLIB_ANNOTATION_BYTES * 2) - p);
          p += n;
        }
        str[p++] = 20;
        str[p++] = 20;
        str[p++] =  0;
      }

      n = edflib_snprint_ll_number_nonlocalized(str + p, annot2->onset / 10000LL, 0, 1, (EDFLIB_ANNOTATION_BYTES * 2) - p);
      p += n;
      if(annot2->onset % 10000LL)
      {
        str[p++] = '.';
        n = edflib_snprint_ll_number_nonlocalized(str + p, annot2->onset % 10000LL, 4, 0, (EDFLIB_ANNOTATION_BYTES * 2) - p);
        p += n;
      }
      if(annot2->duration>=0LL)
      {
        str[p++] = 21;
        n = edflib_snprint_ll_number_nonlocalized(str + p, annot2->duration / 10000LL, 0, 0, (EDFLIB_ANNOTATION_BYTES * 2) - p);
        p += n;
        if(annot2->duration % 10000LL)
        {
          str[p++] = '.';
          n = edflib_snprint_ll_number_nonlocalized(str + p, annot2->duration % 10000LL, 4, 0, (EDFLIB_ANNOTATION_BYTES * 2) - p);
          p += n;
        }
      }
      str[p++] = 20;
      for(i=0; i<EDFLIB_WRITE_MAX_ANNOTATION_LEN; i++)
      {
        if(annot2->annotation[i]==0)
        {
          break;
        }

        str[p++] = annot2->annotation[i];
      }
      str[p++] = 20;

      for(; p<EDFLIB_ANNOTATION_BYTES; p++)
      {
        str[p] = 0;
      }

      nmemb = fwrite(str, EDFLIB_ANNOTATION_BYTES, 1, hdr->file_hdl);

      if(nmemb != 1)
      {
        break;
      }

      j++;

      if(j >= hdr->nr_annot_chns)
      {
        j = 0;

        offset += datrecsize;

        datarecords++;

        if(datarecords>=hdr->datarecords)
        {
          break;
        }
      }
    }

    free(write_annotationslist[handle]);
  }
  else
  {
    free(annotationslist[handle]);
  }

  fclose(hdr->file_hdl);

  free(hdr->edfparam);

  free(hdr->wrbuf);

  free(hdr);

  hdrlist[handle] = NULL;

  edf_files_open--;

  return 0;
}


long long edfseek(int handle, int edfsignal, long long offset, int whence)
{
  long long smp_in_file;

  int channel;


  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  if(edfsignal<0)
  {
    return -1;
  }

  if(hdrlist[handle]->writemode)
  {
    return -1;
  }

  if(edfsignal>=(hdrlist[handle]->edfsignals - hdrlist[handle]->nr_annot_chns))
  {
    return -1;
  }

  channel = hdrlist[handle]->mapped_signals[edfsignal];

  smp_in_file = hdrlist[handle]->edfparam[channel].smp_per_record * hdrlist[handle]->datarecords;

  if(whence==EDFSEEK_SET)
  {
    hdrlist[handle]->edfparam[channel].sample_pntr = offset;
  }

  if(whence==EDFSEEK_CUR)
  {
    hdrlist[handle]->edfparam[channel].sample_pntr += offset;
  }

  if(whence==EDFSEEK_END)
  {
    hdrlist[handle]->edfparam[channel].sample_pntr =
      (hdrlist[handle]->edfparam[channel].smp_per_record * hdrlist[handle]->datarecords) + offset;
  }

  if(hdrlist[handle]->edfparam[channel].sample_pntr > smp_in_file)
  {
    hdrlist[handle]->edfparam[channel].sample_pntr = smp_in_file;
  }

  if(hdrlist[handle]->edfparam[channel].sample_pntr < 0LL)
  {
    hdrlist[handle]->edfparam[channel].sample_pntr = 0LL;
  }

  return hdrlist[handle]->edfparam[channel].sample_pntr;
}


long long edftell(int handle, int edfsignal)
{
  int channel;


  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  if(edfsignal<0)
  {
    return -1;
  }

  if(hdrlist[handle]->writemode)
  {
    return -1;
  }

  if(edfsignal>=(hdrlist[handle]->edfsignals - hdrlist[handle]->nr_annot_chns))
  {
    return -1;
  }

  channel = hdrlist[handle]->mapped_signals[edfsignal];

  return hdrlist[handle]->edfparam[channel].sample_pntr;
}


void edfrewind(int handle, int edfsignal)
{
  int channel;


  if(handle<0)
  {
    return;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return;
  }

  if(hdrlist[handle]==NULL)
  {
    return;
  }

  if(edfsignal<0)
  {
    return;
  }

  if(hdrlist[handle]->writemode)
  {
    return;
  }

  if(edfsignal>=(hdrlist[handle]->edfsignals - hdrlist[handle]->nr_annot_chns))
  {
    return;
  }

  channel = hdrlist[handle]->mapped_signals[edfsignal];

  hdrlist[handle]->edfparam[channel].sample_pntr = 0LL;
}


int edfread_physical_samples(int handle, int edfsignal, int n, double *buf)
{
  int bytes_per_smpl=2,
      tmp,
      i,
      channel;

  double phys_bitvalue,
         phys_offset;

  long long smp_in_file,
            offset,
            sample_pntr,
            smp_per_record,
            jump;

  struct edfhdrblock *hdr;

  union {
          unsigned int one;
          signed int one_signed;
          unsigned short two[2];
          signed short two_signed[2];
          unsigned char four[4];
        } var;

  FILE *file;


  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  if(edfsignal<0)
  {
    return -1;
  }

  if(hdrlist[handle]->writemode)
  {
    return -1;
  }

  if(edfsignal>=(hdrlist[handle]->edfsignals - hdrlist[handle]->nr_annot_chns))
  {
    return -1;
  }

  channel = hdrlist[handle]->mapped_signals[edfsignal];

  if(n<0LL)
  {
    return -1;
  }

  if(n==0LL)
  {
    return 0LL;
  }

  hdr = hdrlist[handle];

  if(hdr->edf)
  {
    bytes_per_smpl = 2;
  }

  if(hdr->bdf)
  {
    bytes_per_smpl = 3;
  }

  smp_in_file = hdr->edfparam[channel].smp_per_record * hdr->datarecords;

  if((hdr->edfparam[channel].sample_pntr + n) > smp_in_file)
  {
    n = smp_in_file - hdr->edfparam[channel].sample_pntr;

    if(n==0)
    {
      return 0LL;
    }

    if(n<0)
    {
      return -1;
    }
  }

  file = hdr->file_hdl;

  offset = hdr->hdrsize;
  offset += (hdr->edfparam[channel].sample_pntr / hdr->edfparam[channel].smp_per_record) * hdr->recordsize;
  offset += hdr->edfparam[channel].buf_offset;
  offset += ((hdr->edfparam[channel].sample_pntr % hdr->edfparam[channel].smp_per_record) * bytes_per_smpl);

  fseeko(file, offset, SEEK_SET);

  sample_pntr = hdr->edfparam[channel].sample_pntr;

  smp_per_record = hdr->edfparam[channel].smp_per_record;

  jump = hdr->recordsize - (smp_per_record * bytes_per_smpl);

  phys_bitvalue = hdr->edfparam[channel].bitvalue;

  phys_offset = hdr->edfparam[channel].offset;

  if(hdr->edf)
  {
    for(i=0; i<n; i++)
    {
      if(!(sample_pntr%smp_per_record))
      {
        if(i)
        {
          fseeko(file, jump, SEEK_CUR);
        }
      }

      var.four[0] = fgetc(file);
      tmp = fgetc(file);
      if(tmp==EOF)
      {
        return -1;
      }
      var.four[1] = tmp;

      buf[i] = phys_bitvalue * (phys_offset + (double)var.two_signed[0]);

      sample_pntr++;
    }
  }

  if(hdr->bdf)
  {
    for(i=0; i<n; i++)
    {
      if(!(sample_pntr%smp_per_record))
      {
        if(i)
        {
          fseeko(file, jump, SEEK_CUR);
        }
      }

      var.four[0] = fgetc(file);
      var.four[1] = fgetc(file);
      tmp = fgetc(file);
      if(tmp==EOF)
      {
        return -1;
      }
      var.four[2] = tmp;

      if(var.four[2]&0x80)
      {
        var.four[3] = 0xff;
      }
      else
      {
        var.four[3] = 0x00;
      }

      buf[i] = phys_bitvalue * (phys_offset + (double)var.one_signed);

      sample_pntr++;
    }
  }

  hdr->edfparam[channel].sample_pntr = sample_pntr;

  return n;
}


int edfread_digital_samples(int handle, int edfsignal, int n, int *buf)
{
  int bytes_per_smpl=2,
      tmp,
      i,
      channel;

  long long smp_in_file,
            offset,
            sample_pntr,
            smp_per_record,
            jump;

  struct edfhdrblock *hdr;

  union {
          unsigned int one;
          signed int one_signed;
          unsigned short two[2];
          signed short two_signed[2];
          unsigned char four[4];
        } var;

  FILE *file;


  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  if(edfsignal<0)
  {
    return -1;
  }

  if(hdrlist[handle]->writemode)
  {
    return -1;
  }

  if(edfsignal>=(hdrlist[handle]->edfsignals - hdrlist[handle]->nr_annot_chns))
  {
    return -1;
  }

  channel = hdrlist[handle]->mapped_signals[edfsignal];

  if(n<0LL)
  {
    return -1;
  }

  if(n==0LL)
  {
    return 0LL;
  }

  hdr = hdrlist[handle];

  if(hdr->edf)
  {
    bytes_per_smpl = 2;
  }

  if(hdr->bdf)
  {
    bytes_per_smpl = 3;
  }

  smp_in_file = hdr->edfparam[channel].smp_per_record * hdr->datarecords;

  if((hdr->edfparam[channel].sample_pntr + n) > smp_in_file)
  {
    n = smp_in_file - hdr->edfparam[channel].sample_pntr;

    if(n==0)
    {
      return 0LL;
    }

    if(n<0)
    {
      return -1;
    }
  }

  file = hdr->file_hdl;

  offset = hdr->hdrsize;
  offset += (hdr->edfparam[channel].sample_pntr / hdr->edfparam[channel].smp_per_record) * hdr->recordsize;
  offset += hdr->edfparam[channel].buf_offset;
  offset += ((hdr->edfparam[channel].sample_pntr % hdr->edfparam[channel].smp_per_record) * bytes_per_smpl);

  fseeko(file, offset, SEEK_SET);

  sample_pntr = hdr->edfparam[channel].sample_pntr;

  smp_per_record = hdr->edfparam[channel].smp_per_record;

  jump = hdr->recordsize - (smp_per_record * bytes_per_smpl);

  if(hdr->edf)
  {
    for(i=0; i<n; i++)
    {
      if(!(sample_pntr%smp_per_record))
      {
        if(i)
        {
          fseeko(file, jump, SEEK_CUR);
        }
      }

      var.four[0] = fgetc(file);
      tmp = fgetc(file);
      if(tmp==EOF)
      {
        return -1;
      }
      var.four[1] = tmp;

      buf[i] = var.two_signed[0];

      sample_pntr++;
    }
  }

  if(hdr->bdf)
  {
    for(i=0; i<n; i++)
    {
      if(!(sample_pntr%smp_per_record))
      {
        if(i)
        {
          fseeko(file, jump, SEEK_CUR);
        }
      }

      var.four[0] = fgetc(file);
      var.four[1] = fgetc(file);
      tmp = fgetc(file);
      if(tmp==EOF)
      {
        return -1;
      }
      var.four[2] = tmp;

      if(var.four[2]&0x80)
      {
        var.four[3] = 0xff;
      }
      else
      {
        var.four[3] = 0x00;
      }

      buf[i] = var.one_signed;

      sample_pntr++;
    }
  }

  hdr->edfparam[channel].sample_pntr = sample_pntr;

  return n;
}


int edf_get_annotation(int handle, int n, struct edf_annotation_struct *annot)
{
  memset(annot, 0, sizeof(struct edf_annotation_struct));

  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  if(hdrlist[handle]->writemode)
  {
    return -1;
  }

  if(n<0)
  {
    return -1;
  }

  if(n>=hdrlist[handle]->annots_in_file)
  {
    return -1;
  }

  annot->onset = (annotationslist[handle] + n)->onset;
  edflib_strlcpy(annot->duration, (annotationslist[handle] + n)->duration, 16);
  edflib_strlcpy(annot->annotation, (annotationslist[handle] + n)->annotation, EDFLIB_MAX_ANNOTATION_LEN + 1);

  return 0;
}


static struct edfhdrblock * edflib_check_edf_file(FILE *inputfile, int *edf_error, int check_file_size)
{
  int i, j, p, r=0, n,
      dotposition,
      error;

  char *edf_hdr,
       scratchpad[128],
       scratchpad2[64];

  struct edfhdrblock *edfhdr;

/***************** check header ******************************/

  edf_hdr = (char *)calloc(1, 256);
  if(edf_hdr==NULL)
  {
    *edf_error = EDFLIB_MALLOC_ERROR;
    return NULL;
  }

  edfhdr = (struct edfhdrblock *)calloc(1, sizeof(struct edfhdrblock));
  if(edfhdr==NULL)
  {
    free(edf_hdr);
    *edf_error = EDFLIB_MALLOC_ERROR;
    return NULL;
  }

  rewind(inputfile);
  if(fread(edf_hdr, 256, 1, inputfile)!=1)
  {
    *edf_error = EDFLIB_FILE_READ_ERROR;
    free(edf_hdr);
    free(edfhdr);
    return NULL;
  }

/**************************** VERSION ***************************************/

  strncpy(scratchpad, edf_hdr, 8);
  scratchpad[8] = 0;

  if(((signed char *)scratchpad)[0]==-1)   /* BDF-file */
  {
    for(i=1; i<8; i++)
    {
      if((scratchpad[i]<32)||(scratchpad[i]>126))
      {
        *edf_error = EDFLIB_FILE_CONTAINS_FORMAT_ERRORS;
        free(edf_hdr);
        free(edfhdr);
        return NULL;
      }
    }

    if(strcmp(scratchpad + 1, "BIOSEMI"))
    {
      *edf_error = EDFLIB_FILE_CONTAINS_FORMAT_ERRORS;
      free(edf_hdr);
      free(edfhdr);
      return NULL;
    }

    edfhdr->bdf = 1;
  }
  else    /* EDF-file */
  {
    for(i=0; i<8; i++)
    {
      if((scratchpad[i]<32)||(scratchpad[i]>126))
      {
        *edf_error = EDFLIB_FILE_CONTAINS_FORMAT_ERRORS;
        free(edf_hdr);
        free(edfhdr);
        return NULL;
      }
    }

    if(strcmp(scratchpad, "0       "))
    {
      *edf_error = EDFLIB_FILE_CONTAINS_FORMAT_ERRORS;
      free(edf_hdr);
      free(edfhdr);
      return NULL;
    }

    edfhdr->edf = 1;
  }

  strncpy(edfhdr->version, edf_hdr, 8);
  edfhdr->version[8] = 0;
  if(edfhdr->bdf)  edfhdr->version[0] = '.';

/********************* PATIENTNAME *********************************************/

  strncpy(scratchpad, edf_hdr + 8, 80);
  scratchpad[80] = 0;
  for(i=0; i<80; i++)
  {
    if((((unsigned char *)scratchpad)[i]<32)||(((unsigned char *)scratchpad)[i]>126))
    {
      *edf_error = EDFLIB_FILE_CONTAINS_FORMAT_ERRORS;
      free(edf_hdr);
      free(edfhdr);
      return NULL;
    }
  }

  strncpy(edfhdr->patient, edf_hdr + 8, 80);
  edfhdr->patient[80] = 0;

/********************* RECORDING *********************************************/

  strncpy(scratchpad, edf_hdr + 88, 80);
  scratchpad[80] = 0;
  for(i=0; i<80; i++)
  {
    if((((unsigned char *)scratchpad)[i]<32)||(((unsigned char *)scratchpad)[i]>126))
    {
      *edf_error = EDFLIB_FILE_CONTAINS_FORMAT_ERRORS;
      free(edf_hdr);
      free(edfhdr);
      return NULL;
    }
  }

  strncpy(edfhdr->recording, edf_hdr + 88, 80);
  edfhdr->recording[80] = 0;

/********************* STARTDATE *********************************************/

  strncpy(scratchpad, edf_hdr + 168, 8);
  scratchpad[8] = 0;
  for(i=0; i<8; i++)
  {
    if((scratchpad[i]<32)||(scratchpad[i]>126))
    {
      *edf_error = EDFLIB_FILE_ERRORS_STARTDATE;
      free(edf_hdr);
      free(edfhdr);
      return NULL;
    }
  }

  error = 0;

  if((edf_hdr[170]!='.')||(edf_hdr[173]!='.'))  error = 1;
  if((edf_hdr[168]<48)||(edf_hdr[168]>57))      error = 1;
  if((edf_hdr[169]<48)||(edf_hdr[169]>57))      error = 1;
  if((edf_hdr[171]<48)||(edf_hdr[171]>57))      error = 1;
  if((edf_hdr[172]<48)||(edf_hdr[172]>57))      error = 1;
  if((edf_hdr[174]<48)||(edf_hdr[174]>57))      error = 1;
  if((edf_hdr[175]<48)||(edf_hdr[175]>57))      error = 1;
  strncpy(scratchpad, edf_hdr + 168, 8);

  if(error)
  {
    scratchpad[8] = 0;
    *edf_error = EDFLIB_FILE_ERRORS_STARTDATE;
    free(edf_hdr);
    free(edfhdr);
    return NULL;
  }

  scratchpad[2] = 0;
  scratchpad[5] = 0;
  scratchpad[8] = 0;

  if((edflib_atof_nonlocalized(scratchpad)<1)||(edflib_atof_nonlocalized(scratchpad)>31))
  {
    strncpy(scratchpad, edf_hdr + 168, 8);
    scratchpad[8] = 0;
    *edf_error = EDFLIB_FILE_ERRORS_STARTDATE;
    free(edf_hdr);
    free(edfhdr);
    return NULL;
  }

  if((edflib_atof_nonlocalized(scratchpad+3)<1)||(edflib_atof_nonlocalized(scratchpad+3)>12))
  {
    strncpy(scratchpad, edf_hdr + 168, 8);
    scratchpad[8] = 0;
    *edf_error = EDFLIB_FILE_ERRORS_STARTDATE;
    free(edf_hdr);
    free(edfhdr);
    return NULL;
  }

  edfhdr->startdate_day = edflib_atof_nonlocalized(scratchpad);
  edfhdr->startdate_month = edflib_atof_nonlocalized(scratchpad + 3);
  edfhdr->startdate_year = edflib_atof_nonlocalized(scratchpad + 6);
  if(edfhdr->startdate_year>84)
  {
    edfhdr->startdate_year += 1900;
  }
  else
  {
    edfhdr->startdate_year += 2000;
  }

/********************* STARTTIME *********************************************/

  strncpy(scratchpad, edf_hdr + 176, 8);
  scratchpad[8] = 0;
  for(i=0; i<8; i++)
  {
    if((scratchpad[i]<32)||(scratchpad[i]>126))
    {
      *edf_error = EDFLIB_FILE_ERRORS_STARTTIME;
      free(edf_hdr);
      free(edfhdr);
      return NULL;
    }
  }

  error = 0;

  if((edf_hdr[178]!='.')||(edf_hdr[181]!='.'))  error = 1;
  if((edf_hdr[176]<48)||(edf_hdr[176]>57))      error = 1;
  if((edf_hdr[177]<48)||(edf_hdr[177]>57))      error = 1;
  if((edf_hdr[179]<48)||(edf_hdr[179]>57))      error = 1;
  if((edf_hdr[180]<48)||(edf_hdr[180]>57))      error = 1;
  if((edf_hdr[182]<48)||(edf_hdr[182]>57))      error = 1;
  if((edf_hdr[183]<48)||(edf_hdr[183]>57))      error = 1;

  strncpy(scratchpad, edf_hdr + 176, 8);

  if(error)
  {
    strncpy(scratchpad, edf_hdr + 176, 8);
    scratchpad[8] = 0;
    *edf_error = EDFLIB_FILE_ERRORS_STARTTIME;
    free(edf_hdr);
    free(edfhdr);
    return NULL;
  }

  scratchpad[2] = 0;
  scratchpad[5] = 0;
  scratchpad[8] = 0;

  if(edflib_atof_nonlocalized(scratchpad)>23)
  {
    strncpy(scratchpad, edf_hdr + 176, 8);
    scratchpad[8] = 0;
    *edf_error = EDFLIB_FILE_ERRORS_STARTTIME;
    free(edf_hdr);
    free(edfhdr);
    return NULL;
  }

  if(edflib_atof_nonlocalized(scratchpad+3)>59)
  {
    strncpy(scratchpad, edf_hdr + 176, 8);
    scratchpad[8] = 0;
    *edf_error = EDFLIB_FILE_ERRORS_STARTTIME;
    free(edf_hdr);
    free(edfhdr);
    return NULL;
  }

  if(edflib_atof_nonlocalized(scratchpad+6)>59)
  {
    strncpy(scratchpad, edf_hdr + 176, 8);
    scratchpad[8] = 0;
    *edf_error = EDFLIB_FILE_ERRORS_STARTTIME;
    free(edf_hdr);
    free(edfhdr);
    return NULL;
  }

  edfhdr->starttime_hour = edflib_atof_nonlocalized(scratchpad);
  edfhdr->starttime_minute = edflib_atof_nonlocalized(scratchpad + 3);
  edfhdr->starttime_second = edflib_atof_nonlocalized(scratchpad + 6);

  edfhdr->l_starttime = 3600 * edflib_atof_nonlocalized(scratchpad);
  edfhdr->l_starttime += 60 * edflib_atof_nonlocalized(scratchpad + 3);
  edfhdr->l_starttime += edflib_atof_nonlocalized(scratchpad + 6);

  edfhdr->l_starttime *= EDFLIB_TIME_DIMENSION;

/***************** NUMBER OF SIGNALS IN HEADER *******************************/

  strncpy(scratchpad, edf_hdr + 252, 4);
  scratchpad[4] = 0;
  for(i=0; i<4; i++)
  {
    if((scratchpad[i]<32)||(scratchpad[i]>126))
    {
      *edf_error = EDFLIB_FILE_ERRORS_NUMBER_SIGNALS;
      free(edf_hdr);
      free(edfhdr);
      return NULL;
    }
  }

  if(edflib_is_integer_number(scratchpad))
  {
    *edf_error = EDFLIB_FILE_ERRORS_NUMBER_SIGNALS;
    free(edf_hdr);
    free(edfhdr);
    return NULL;
  }
  edfhdr->edfsignals = edflib_atof_nonlocalized(scratchpad);
  if(edfhdr->edfsignals<1)
  {
    *edf_error = EDFLIB_FILE_ERRORS_NUMBER_SIGNALS;
    free(edf_hdr);
    free(edfhdr);
    return NULL;
  }

  if(edfhdr->edfsignals>EDFLIB_MAXSIGNALS)
  {
    *edf_error = EDFLIB_FILE_ERRORS_NUMBER_SIGNALS;
    free(edf_hdr);
    free(edfhdr);
    return NULL;
  }

/***************** NUMBER OF BYTES IN HEADER *******************************/

  strncpy(scratchpad, edf_hdr + 184, 8);
  scratchpad[8] = 0;

  for(i=0; i<8; i++)
  {
    if((scratchpad[i]<32)||(scratchpad[i]>126))
    {
      *edf_error = EDFLIB_FILE_ERRORS_BYTES_HEADER;
      free(edf_hdr);
      free(edfhdr);
      return NULL;
    }
  }

  if(edflib_is_integer_number(scratchpad))
  {
    *edf_error = EDFLIB_FILE_ERRORS_BYTES_HEADER;
    free(edf_hdr);
    free(edfhdr);
    return NULL;
  }

  n  = edflib_atof_nonlocalized(scratchpad);
  if((edfhdr->edfsignals * 256 + 256)!=n)
  {
    *edf_error = EDFLIB_FILE_ERRORS_BYTES_HEADER;
    free(edf_hdr);
    free(edfhdr);
    return NULL;
  }

/********************* RESERVED FIELD *************************************/

  edfhdr->edfplus = 0;
  edfhdr->discontinuous = 0;
  strncpy(scratchpad, edf_hdr + 192, 44);
  scratchpad[44] = 0;

  for(i=0; i<44; i++)
  {
    if((scratchpad[i]<32)||(scratchpad[i]>126))
    {
      *edf_error = EDFLIB_FILE_ERRORS_RESERVED_FIELD;
      free(edf_hdr);
      free(edfhdr);
      return NULL;
    }
  }

  if(edfhdr->edf)
  {
    if(!strncmp(scratchpad, "EDF+C", 5))
    {
      edfhdr->edfplus = 1;
    }

    if(!strncmp(scratchpad, "EDF+D", 5))
    {
      edfhdr->edfplus = 1;
      edfhdr->discontinuous = 1;
    }
  }

  if(edfhdr->bdf)
  {
    if(!strncmp(scratchpad, "BDF+C", 5))
    {
      edfhdr->bdfplus = 1;
    }

    if(!strncmp(scratchpad, "BDF+D", 5))
    {
      edfhdr->bdfplus = 1;
      edfhdr->discontinuous = 1;
    }
  }

  strncpy(edfhdr->reserved, edf_hdr + 192, 44);
  edfhdr->reserved[44] = 0;

/********************* NUMBER OF DATARECORDS *************************************/

  strncpy(scratchpad, edf_hdr + 236, 8);
  scratchpad[8] = 0;

  for(i=0; i<8; i++)
  {
    if((scratchpad[i]<32)||(scratchpad[i]>126))
    {
      *edf_error = EDFLIB_FILE_ERRORS_NUMBER_DATARECORDS;
      free(edf_hdr);
      free(edfhdr);
      return NULL;
    }
  }

  if(edflib_is_integer_number(scratchpad))
  {
    *edf_error = EDFLIB_FILE_ERRORS_NUMBER_DATARECORDS;
    free(edf_hdr);
    free(edfhdr);
    return NULL;
  }

  edfhdr->datarecords = edflib_atof_nonlocalized(scratchpad);
  if(edfhdr->datarecords<1)
  {
    *edf_error = EDFLIB_FILE_ERRORS_NUMBER_DATARECORDS;
    free(edf_hdr);
    free(edfhdr);
    return NULL;
  }

/********************* DATARECORD DURATION *************************************/

  strncpy(scratchpad, edf_hdr + 244, 8);
  scratchpad[8] = 0;

  for(i=0; i<8; i++)
  {
    if((scratchpad[i]<32)||(scratchpad[i]>126))
    {
      *edf_error = EDFLIB_FILE_ERRORS_DURATION;
      free(edf_hdr);
      free(edfhdr);
      return NULL;
    }
  }

  if(edflib_is_number(scratchpad))
  {
    *edf_error = EDFLIB_FILE_ERRORS_DURATION;
    free(edf_hdr);
    free(edfhdr);
    return NULL;
  }

  edfhdr->data_record_duration = edflib_atof_nonlocalized(scratchpad);
  if(edfhdr->data_record_duration < -0.000001)
  {
    *edf_error = EDFLIB_FILE_ERRORS_DURATION;
    free(edf_hdr);
    free(edfhdr);
    return NULL;
  }

  edfhdr->long_data_record_duration = edflib_get_long_duration(scratchpad);

  free(edf_hdr);

/********************* START WITH THE SIGNALS IN THE HEADER *********************/

  edf_hdr = (char *)calloc(1, (edfhdr->edfsignals + 1) * 256);
  if(edf_hdr==NULL)
  {
    *edf_error = EDFLIB_MALLOC_ERROR;
    free(edfhdr);
    return NULL;
  }

  rewind(inputfile);
  if(fread(edf_hdr, (edfhdr->edfsignals + 1) * 256, 1, inputfile)!=1)
  {
    *edf_error = EDFLIB_FILE_READ_ERROR;
    free(edf_hdr);
    free(edfhdr);
    return NULL;
  }

  edfhdr->edfparam = (struct edfparamblock *)calloc(1, sizeof(struct edfparamblock) * edfhdr->edfsignals);
  if(edfhdr->edfparam==NULL)
  {
    *edf_error = EDFLIB_MALLOC_ERROR;
    free(edf_hdr);
    free(edfhdr);
    return NULL;
  }

/**************************** LABELS *************************************/

  edfhdr->nr_annot_chns = 0;
  for(i=0; i<edfhdr->edfsignals; i++)
  {
    strncpy(scratchpad, edf_hdr + 256 + (i * 16), 16);
    for(j=0; j<16; j++)
    {
      if((scratchpad[j]<32)||(scratchpad[j]>126))
      {
        *edf_error = EDFLIB_FILE_ERRORS_LABEL;
        free(edf_hdr);
        free(edfhdr->edfparam);
        free(edfhdr);
        return NULL;
      }
    }
    if(edfhdr->edfplus)
    {
      if(!strncmp(scratchpad, "EDF Annotations ", 16))
      {
        edfhdr->annot_ch[edfhdr->nr_annot_chns] = i;
        edfhdr->nr_annot_chns++;
        edfhdr->edfparam[i].annotation = 1;
      }
    }
    if(edfhdr->bdfplus)
    {
      if(!strncmp(scratchpad, "BDF Annotations ", 16))
      {
        edfhdr->annot_ch[edfhdr->nr_annot_chns] = i;
        edfhdr->nr_annot_chns++;
        edfhdr->edfparam[i].annotation = 1;
      }
    }
    strncpy(edfhdr->edfparam[i].label, edf_hdr + 256 + (i * 16), 16);
    edfhdr->edfparam[i].label[16] = 0;
  }
  if(edfhdr->edfplus&&(!edfhdr->nr_annot_chns))
  {
    *edf_error = EDFLIB_FILE_ERRORS_LABEL;
    free(edf_hdr);
    free(edfhdr->edfparam);
    free(edfhdr);
    return NULL;
  }
  if(edfhdr->bdfplus&&(!edfhdr->nr_annot_chns))
  {
    *edf_error = EDFLIB_FILE_ERRORS_LABEL;
    free(edf_hdr);
    free(edfhdr->edfparam);
    free(edfhdr);
    return NULL;
  }
  if((edfhdr->edfsignals!=edfhdr->nr_annot_chns)||((!edfhdr->edfplus)&&(!edfhdr->bdfplus)))
  {
	//if(edfhdr->data_record_duration<0.0000001)
    if(edfhdr->data_record_duration<0)
    {
      *edf_error = EDFLIB_FILE_ERRORS_LABEL;
      free(edf_hdr);
      free(edfhdr->edfparam);
      free(edfhdr);
      return NULL;
    }
  }

/**************************** TRANSDUCER TYPES *************************************/

  for(i=0; i<edfhdr->edfsignals; i++)
  {
    strncpy(scratchpad, edf_hdr + 256 + (edfhdr->edfsignals * 16) + (i * 80), 80);
    for(j=0; j<80; j++)
    {
      if((scratchpad[j]<32)||(scratchpad[j]>126))
      {
        *edf_error = EDFLIB_FILE_ERRORS_TRANSDUCER;
        free(edf_hdr);
        free(edfhdr->edfparam);
        free(edfhdr);
        return NULL;
      }
    }
    strncpy(edfhdr->edfparam[i].transducer, edf_hdr + 256 + (edfhdr->edfsignals * 16) + (i * 80), 80);
    edfhdr->edfparam[i].transducer[80] = 0;

    if((edfhdr->edfplus) || (edfhdr->bdfplus))
    {
      if(edfhdr->edfparam[i].annotation)
      {
        for(j=0; j<80; j++)
        {
          if(edfhdr->edfparam[i].transducer[j]!=' ')
          {
            *edf_error = EDFLIB_FILE_ERRORS_TRANSDUCER;
            free(edf_hdr);
            free(edfhdr->edfparam);
            free(edfhdr);
            return NULL;
          }
        }
      }
    }
  }

/**************************** PHYSICAL DIMENSIONS *************************************/

  for(i=0; i<edfhdr->edfsignals; i++)
  {
    strncpy(scratchpad, edf_hdr + 256 + (edfhdr->edfsignals * 96) + (i * 8), 8);
    for(j=0; j<8; j++)
    {
      if((scratchpad[j]<32)||(scratchpad[j]>126))
      {
        *edf_error = EDFLIB_FILE_ERRORS_PHYS_DIMENSION;
        free(edf_hdr);
        free(edfhdr->edfparam);
        free(edfhdr);
        return NULL;
      }
    }
    strncpy(edfhdr->edfparam[i].physdimension, edf_hdr + 256 + (edfhdr->edfsignals * 96) + (i * 8), 8);
    edfhdr->edfparam[i].physdimension[8] = 0;
  }

/**************************** PHYSICAL MINIMUMS *************************************/

  for(i=0; i<edfhdr->edfsignals; i++)
  {
    strncpy(scratchpad, edf_hdr + 256 + (edfhdr->edfsignals * 104) + (i * 8), 8);
    scratchpad[8] = 0;

    for(j=0; j<8; j++)
    {
      if((scratchpad[j]<32)||(scratchpad[j]>126))
      {
        *edf_error = EDFLIB_FILE_ERRORS_PHYS_MIN;
        free(edf_hdr);
        free(edfhdr->edfparam);
        free(edfhdr);
        return NULL;
      }
    }

    if(edflib_is_number(scratchpad))
    {
      *edf_error = EDFLIB_FILE_ERRORS_PHYS_MIN;
      free(edf_hdr);
      free(edfhdr->edfparam);
      free(edfhdr);
      return NULL;
    }

    edfhdr->edfparam[i].phys_min = edflib_atof_nonlocalized(scratchpad);
  }

/**************************** PHYSICAL MAXIMUMS *************************************/

  for(i=0; i<edfhdr->edfsignals; i++)
  {
    strncpy(scratchpad, edf_hdr + 256 + (edfhdr->edfsignals * 112) + (i * 8), 8);
    scratchpad[8] = 0;

    for(j=0; j<8; j++)
    {
      if((scratchpad[j]<32)||(scratchpad[j]>126))
      {
        *edf_error = EDFLIB_FILE_ERRORS_PHYS_MAX;
        free(edf_hdr);
        free(edfhdr->edfparam);
        free(edfhdr);
        return NULL;
      }
    }

    if(edflib_is_number(scratchpad))
    {
      *edf_error = EDFLIB_FILE_ERRORS_PHYS_MAX;
      free(edf_hdr);
      free(edfhdr->edfparam);
      free(edfhdr);
      return NULL;
    }

    edfhdr->edfparam[i].phys_max = edflib_atof_nonlocalized(scratchpad);
    if(edfhdr->edfparam[i].phys_max==edfhdr->edfparam[i].phys_min)
    {
      *edf_error = EDFLIB_FILE_ERRORS_PHYS_MAX;
      free(edf_hdr);
      free(edfhdr->edfparam);
      free(edfhdr);
      return NULL;
    }
  }

/**************************** DIGITAL MINIMUMS *************************************/

  for(i=0; i<edfhdr->edfsignals; i++)
  {
    strncpy(scratchpad, edf_hdr + 256 + (edfhdr->edfsignals * 120) + (i * 8), 8);
    scratchpad[8] = 0;

    for(j=0; j<8; j++)
    {
      if((scratchpad[j]<32)||(scratchpad[j]>126))
      {
        *edf_error = EDFLIB_FILE_ERRORS_DIG_MIN;
        free(edf_hdr);
        free(edfhdr->edfparam);
        free(edfhdr);
        return NULL;
      }
    }

    if(edflib_is_integer_number(scratchpad))
    {
      *edf_error = EDFLIB_FILE_ERRORS_DIG_MIN;
      free(edf_hdr);
      free(edfhdr->edfparam);
      free(edfhdr);
      return NULL;
    }

    n = edflib_atof_nonlocalized(scratchpad);
    if(edfhdr->edfplus)
    {
      if(edfhdr->edfparam[i].annotation)
      {
        if(n!=-32768)
        {
          *edf_error = EDFLIB_FILE_ERRORS_DIG_MIN;
          free(edf_hdr);
          free(edfhdr->edfparam);
          free(edfhdr);
          return NULL;
        }
      }
    }
    if(edfhdr->bdfplus)
    {
      if(edfhdr->edfparam[i].annotation)
      {
        if(n!=-8388608)
        {
          *edf_error = EDFLIB_FILE_ERRORS_DIG_MIN;
          free(edf_hdr);
          free(edfhdr->edfparam);
          free(edfhdr);
          return NULL;
        }
      }
    }
    if(edfhdr->edf)
    {
      if((n>32767)||(n<-32768))
      {
        *edf_error = EDFLIB_FILE_ERRORS_DIG_MIN;
        free(edf_hdr);
        free(edfhdr->edfparam);
        free(edfhdr);
        return NULL;
      }
    }
    if(edfhdr->bdf)
    {
      if((n>8388607)||(n<-8388608))
      {
        *edf_error = EDFLIB_FILE_ERRORS_DIG_MIN;
        free(edf_hdr);
        free(edfhdr->edfparam);
        free(edfhdr);
        return NULL;
      }
    }
    edfhdr->edfparam[i].dig_min = n;
  }

/**************************** DIGITAL MAXIMUMS *************************************/

  for(i=0; i<edfhdr->edfsignals; i++)
  {
    strncpy(scratchpad, edf_hdr + 256 + (edfhdr->edfsignals * 128) + (i * 8), 8);
    scratchpad[8] = 0;

    for(j=0; j<8; j++)
    {
      if((scratchpad[j]<32)||(scratchpad[j]>126))
      {
        *edf_error = EDFLIB_FILE_ERRORS_DIG_MAX;
        free(edf_hdr);
        free(edfhdr->edfparam);
        free(edfhdr);
        return NULL;
      }
    }

    if(edflib_is_integer_number(scratchpad))
    {
      *edf_error = EDFLIB_FILE_ERRORS_DIG_MAX;
      free(edf_hdr);
      free(edfhdr->edfparam);
      free(edfhdr);
      return NULL;
    }

    n = edflib_atof_nonlocalized(scratchpad);
    if(edfhdr->edfplus)
    {
      if(edfhdr->edfparam[i].annotation)
      {
        if(n!=32767)
        {
          *edf_error = EDFLIB_FILE_ERRORS_DIG_MAX;
          free(edf_hdr);
          free(edfhdr->edfparam);
          free(edfhdr);
          return NULL;
        }
      }
    }
    if(edfhdr->bdfplus)
    {
      if(edfhdr->edfparam[i].annotation)
      {
        if(n!=8388607)
        {
          *edf_error = EDFLIB_FILE_ERRORS_DIG_MAX;
          free(edf_hdr);
          free(edfhdr->edfparam);
          free(edfhdr);
          return NULL;
        }
      }
    }
    if(edfhdr->edf)
    {
      if((n>32767)||(n<-32768))
      {
        *edf_error = EDFLIB_FILE_ERRORS_DIG_MAX;
        free(edf_hdr);
        free(edfhdr->edfparam);
        free(edfhdr);
        return NULL;
      }
    }
    else
    {
      if((n>8388607)||(n<-8388608))
      {
        *edf_error = EDFLIB_FILE_ERRORS_DIG_MAX;
        free(edf_hdr);
        free(edfhdr->edfparam);
        free(edfhdr);
        return NULL;
      }
    }
    edfhdr->edfparam[i].dig_max = n;
    if(edfhdr->edfparam[i].dig_max<(edfhdr->edfparam[i].dig_min + 1) && (edfhdr->bdfplus || edfhdr->edfplus))
    {
      *edf_error = EDFLIB_FILE_ERRORS_DIG_MAX;
      free(edf_hdr);
      free(edfhdr->edfparam);
      free(edfhdr);
      return NULL;
    }
  }

/**************************** PREFILTER FIELDS *************************************/

  for(i=0; i<edfhdr->edfsignals; i++)
  {
    strncpy(scratchpad, edf_hdr + 256 + (edfhdr->edfsignals * 136) + (i * 80), 80);
    for(j=0; j<80; j++)
    {
      if((scratchpad[j]<32)||(scratchpad[j]>126))
      {
        *edf_error = EDFLIB_FILE_ERRORS_PREFILTER;
        free(edf_hdr);
        free(edfhdr->edfparam);
        free(edfhdr);
        return NULL;
      }
    }
    strncpy(edfhdr->edfparam[i].prefilter, edf_hdr + 256 + (edfhdr->edfsignals * 136) + (i * 80), 80);
    edfhdr->edfparam[i].prefilter[80] = 0;
  }

/*********************** NR OF SAMPLES IN EACH DATARECORD ********************/

  edfhdr->recordsize = 0;

  for(i=0; i<edfhdr->edfsignals; i++)
  {
    strncpy(scratchpad, edf_hdr + 256 + (edfhdr->edfsignals * 216) + (i * 8), 8);
    scratchpad[8] = 0;

    for(j=0; j<8; j++)
    {
      if((scratchpad[j]<32)||(scratchpad[j]>126))
      {
        *edf_error = EDFLIB_FILE_ERRORS_SAMPLES_DATARECORD;
        free(edf_hdr);
        free(edfhdr->edfparam);
        free(edfhdr);
        return NULL;
      }
    }

    if(edflib_is_integer_number(scratchpad))
    {
      *edf_error = EDFLIB_FILE_ERRORS_SAMPLES_DATARECORD;
      free(edf_hdr);
      free(edfhdr->edfparam);
      free(edfhdr);
      return NULL;
    }

    n = edflib_atof_nonlocalized(scratchpad);
    if(n<1)
    {
      *edf_error = EDFLIB_FILE_ERRORS_SAMPLES_DATARECORD;
      free(edf_hdr);
      free(edfhdr->edfparam);
      free(edfhdr);
      return NULL;
    }
    edfhdr->edfparam[i].smp_per_record = n;
    edfhdr->recordsize += n;
  }

  if(edfhdr->bdf)
  {
    edfhdr->recordsize *= 3;

    if(edfhdr->recordsize > (15 * 1024 * 1024))
    {
      *edf_error = EDFLIB_FILE_ERRORS_SAMPLES_DATARECORD;
      free(edf_hdr);
      free(edfhdr->edfparam);
      free(edfhdr);
      return NULL;
    }
  }
  else
  {
    edfhdr->recordsize *= 2;

    if(edfhdr->recordsize > (10 * 1024 * 1024))
    {
      *edf_error = EDFLIB_FILE_ERRORS_SAMPLES_DATARECORD;
      free(edf_hdr);
      free(edfhdr->edfparam);
      free(edfhdr);
      return NULL;
    }
  }

/**************************** RESERVED FIELDS *************************************/

  for(i=0; i<edfhdr->edfsignals; i++)
  {
    strncpy(scratchpad, edf_hdr + 256 + (edfhdr->edfsignals * 224) + (i * 32), 32);
    for(j=0; j<32; j++)
    {
      if((scratchpad[j]<32)||(scratchpad[j]>126))
      {
        *edf_error = EDFLIB_FILE_CONTAINS_FORMAT_ERRORS;
        free(edf_hdr);
        free(edfhdr->edfparam);
        free(edfhdr);
        return NULL;
      }
    }
    strncpy(edfhdr->edfparam[i].reserved, edf_hdr + 256 + (edfhdr->edfsignals * 224) + (i * 32), 32);
    edfhdr->edfparam[i].reserved[32] = 0;
  }

/********************* EDF+ PATIENTNAME *********************************************/

  if(edfhdr->edfplus || edfhdr->bdfplus)
  {
    error = 0;
    dotposition = 0;
    strncpy(scratchpad, edf_hdr + 8, 80);
    scratchpad[80] = 0;
    for(i=0; i<80; i++)
    {
      if(scratchpad[i]==' ')
      {
        dotposition = i;
        break;
      }
    }
    dotposition++;
    if((dotposition>73)||(dotposition<2))  error = 1;
    if(scratchpad[dotposition + 2]!='X')
    {
      if(dotposition>65)  error = 1;
    }
    if((scratchpad[dotposition]!='M')&&(scratchpad[dotposition]!='F')&&(scratchpad[dotposition]!='X'))  error = 1;
    dotposition++;
    if(scratchpad[dotposition]!=' ')  error = 1;
    if(scratchpad[dotposition + 1]=='X')
    {
      if(scratchpad[dotposition + 2]!=' ')  error = 1;
      if(scratchpad[dotposition + 3]==' ')  error = 1;
    }
    else
    {
      if(scratchpad[dotposition + 12]!=' ')  error = 1;
      if(scratchpad[dotposition + 13]==' ')  error = 1;
      dotposition++;
      strncpy(scratchpad2, scratchpad + dotposition, 11);
      scratchpad2[11] = 0;
      if((scratchpad2[2]!='-')||(scratchpad2[6]!='-'))  error = 1;
      scratchpad2[2] = 0;
      scratchpad2[6] = 0;
      if((scratchpad2[0]<48)||(scratchpad2[0]>57))  error = 1;
      if((scratchpad2[1]<48)||(scratchpad2[1]>57))  error = 1;
      if((scratchpad2[7]<48)||(scratchpad2[7]>57))  error = 1;
      if((scratchpad2[8]<48)||(scratchpad2[8]>57))  error = 1;
      if((scratchpad2[9]<48)||(scratchpad2[9]>57))  error = 1;
      if((scratchpad2[10]<48)||(scratchpad2[10]>57))  error = 1;
      if((edflib_atof_nonlocalized(scratchpad2)<1)||(edflib_atof_nonlocalized(scratchpad2)>31))  error = 1;
      if(strcmp(scratchpad2 + 3, "JAN"))
        if(strcmp(scratchpad2 + 3, "FEB"))
          if(strcmp(scratchpad2 + 3, "MAR"))
            if(strcmp(scratchpad2 + 3, "APR"))
              if(strcmp(scratchpad2 + 3, "MAY"))
                if(strcmp(scratchpad2 + 3, "JUN"))
                  if(strcmp(scratchpad2 + 3, "JUL"))
                    if(strcmp(scratchpad2 + 3, "AUG"))
                      if(strcmp(scratchpad2 + 3, "SEP"))
                        if(strcmp(scratchpad2 + 3, "OCT"))
                          if(strcmp(scratchpad2 + 3, "NOV"))
                            if(strcmp(scratchpad2 + 3, "DEC"))
                              error = 1;
    }

    if(error)
    {
      *edf_error = EDFLIB_FILE_CONTAINS_FORMAT_ERRORS;
      free(edf_hdr);
      free(edfhdr->edfparam);
      free(edfhdr);
      return NULL;
    }

    p = 0;
    if((edfhdr->patient[p]=='X') && (edfhdr->patient[p+1]==' '))
    {
      edfhdr->plus_patientcode[0] = 0;
      p += 2;
    }
    else
    {
      for(i=0; i<(80-p); i++)
      {
        if(edfhdr->patient[i+p]==' ')
        {
          break;
        }
        edfhdr->plus_patientcode[i] = edfhdr->patient[i+p];
        if(edfhdr->plus_patientcode[i]=='_')  edfhdr->plus_patientcode[i] = ' ';
      }
      edfhdr->plus_patientcode[i] = 0;
      p += i + 1;
    }

    if(edfhdr->patient[p]=='M')
    {
      edflib_strlcpy(edfhdr->plus_gender, "Male", 16);
    }
    if(edfhdr->patient[p]=='F')
    {
      edflib_strlcpy(edfhdr->plus_gender, "Female", 16);
    }
    if(edfhdr->patient[p]=='X')
    {
      edfhdr->plus_gender[0] = 0;
    }
    for(i=0; i<(80-p);i++)
    {
      if(edfhdr->patient[i+p]==' ')
      {
        break;
      }
    }
    p += i + 1;

    if(edfhdr->patient[p]=='X')
    {
      edfhdr->plus_birthdate[0] = 0;
      p += 2;
    }
    else
    {
      for(i=0; i<(80-p); i++)
      {
        if(edfhdr->patient[i+p]==' ')
        {
          break;
        }
        edfhdr->plus_birthdate[i] = edfhdr->patient[i+p];
      }
      edfhdr->plus_birthdate[2] = ' ';
      edfhdr->plus_birthdate[3] += 32;
      edfhdr->plus_birthdate[4] += 32;
      edfhdr->plus_birthdate[5] += 32;
      edfhdr->plus_birthdate[6] = ' ';
      edfhdr->plus_birthdate[11] = 0;
      p += i + 1;
    }

    for(i=0; i<(80-p);i++)
    {
      if(edfhdr->patient[i+p]==' ')
      {
        break;
      }
      edfhdr->plus_patient_name[i] = edfhdr->patient[i+p];
      if(edfhdr->plus_patient_name[i]=='_')  edfhdr->plus_patient_name[i] = ' ';
    }
    edfhdr->plus_patient_name[i] = 0;
    p += i + 1;

    for(i=0; i<(80-p);i++)
    {
      edfhdr->plus_patient_additional[i] = edfhdr->patient[i+p];
    }
    edfhdr->plus_patient_additional[i] = 0;
    p += i + 1;
  }

/********************* EDF+ RECORDINGFIELD *********************************************/

  if(edfhdr->edfplus || edfhdr->bdfplus)
  {
    error = 0;
    strncpy(scratchpad, edf_hdr + 88, 80);
    scratchpad[80] = 0;
    if(strncmp(scratchpad, "Startdate ", 10))  error = 1;
    if(scratchpad[10]=='X')
    {
      if(scratchpad[11]!=' ')  error = 1;
      if(scratchpad[12]==' ')  error = 1;
      p = 12;
    }
    else
    {
      if(scratchpad[21]!=' ')  error = 1;
      if(scratchpad[22]==' ')  error = 1;
      p = 22;
      strncpy(scratchpad2, scratchpad + 10, 11);
      scratchpad2[11] = 0;
      if((scratchpad2[2]!='-')||(scratchpad2[6]!='-'))  error = 1;
      scratchpad2[2] = 0;
      scratchpad2[6] = 0;
      if((scratchpad2[0]<48)||(scratchpad2[0]>57))  error = 1;
      if((scratchpad2[1]<48)||(scratchpad2[1]>57))  error = 1;
      if((scratchpad2[7]<48)||(scratchpad2[7]>57))  error = 1;
      if((scratchpad2[8]<48)||(scratchpad2[8]>57))  error = 1;
      if((scratchpad2[9]<48)||(scratchpad2[9]>57))  error = 1;
      if((scratchpad2[10]<48)||(scratchpad2[10]>57))  error = 1;
      if((edflib_atof_nonlocalized(scratchpad2)<1)||(edflib_atof_nonlocalized(scratchpad2)>31))  error = 1;
      r = 0;
      if(!strcmp(scratchpad2 + 3, "JAN"))  r = 1;
        else if(!strcmp(scratchpad2 + 3, "FEB"))  r = 2;
          else if(!strcmp(scratchpad2 + 3, "MAR"))  r = 3;
            else if(!strcmp(scratchpad2 + 3, "APR"))  r = 4;
              else if(!strcmp(scratchpad2 + 3, "MAY"))  r = 5;
                else if(!strcmp(scratchpad2 + 3, "JUN"))  r = 6;
                  else if(!strcmp(scratchpad2 + 3, "JUL"))  r = 7;
                    else if(!strcmp(scratchpad2 + 3, "AUG"))  r = 8;
                      else if(!strcmp(scratchpad2 + 3, "SEP"))  r = 9;
                        else if(!strcmp(scratchpad2 + 3, "OCT"))  r = 10;
                          else if(!strcmp(scratchpad2 + 3, "NOV"))  r = 11;
                            else if(!strcmp(scratchpad2 + 3, "DEC"))  r = 12;
                              else error = 1;
    }

    n = 0;
    for(i=p; i<80; i++)
    {
      if(i>78)
      {
        error = 1;
        break;
      }
      if(scratchpad[i]==' ')
      {
        n++;
        if(scratchpad[i + 1]==' ')
        {
          error = 1;
          break;
        }
      }
      if(n>1)  break;
    }

    if(error)
    {
      *edf_error = EDFLIB_FILE_ERRORS_RECORDINGFIELD;
      free(edf_hdr);
      free(edfhdr->edfparam);
      free(edfhdr);
      return NULL;
    }

    if(edf_hdr[98]!='X')
    {
      error = 0;

      strncpy(scratchpad, edf_hdr + 168, 8);
      scratchpad[2] = 0;
      scratchpad[5] = 0;
      scratchpad[8] = 0;

      if(edflib_atof_nonlocalized(scratchpad)!=edflib_atof_nonlocalized(scratchpad2))  error = 1;
      if(edflib_atof_nonlocalized(scratchpad+3)!=r)  error = 1;
      if(edflib_atof_nonlocalized(scratchpad+6)!=edflib_atof_nonlocalized(scratchpad2+9))  error = 1;
      if(error)
      {
        *edf_error = EDFLIB_FILE_ERRORS_RECORDINGFIELD;
        free(edf_hdr);
        free(edfhdr->edfparam);
        free(edfhdr);
        return NULL;
      }

      edfhdr->startdate_year = edflib_atof_nonlocalized(scratchpad2 + 7);

      if(edfhdr->startdate_year<1970)
      {
        *edf_error = EDFLIB_FILE_ERRORS_RECORDINGFIELD;
        free(edf_hdr);
        free(edfhdr->edfparam);
        free(edfhdr);
        return NULL;
      }
    }

    p = 10;
    for(i=0; i<(80-p); i++)
    {
      if(edfhdr->recording[i+p]==' ')
      {
        break;
      }
      edfhdr->plus_startdate[i] = edfhdr->recording[i+p];
    }
    edfhdr->plus_startdate[2] = ' ';
    edfhdr->plus_startdate[3] += 32;
    edfhdr->plus_startdate[4] += 32;
    edfhdr->plus_startdate[5] += 32;
    edfhdr->plus_startdate[6] = ' ';
    edfhdr->plus_startdate[11] = 0;
    p += i + 1;

    if((edfhdr->recording[p]=='X') && (edfhdr->recording[p+1]==' '))
    {
      edfhdr->plus_admincode[0] = 0;
      p += 2;
    }
    else
    {
      for(i=0; i<(80-p); i++)
      {
        if(edfhdr->recording[i+p]==' ')
        {
          break;
        }
        edfhdr->plus_admincode[i] = edfhdr->recording[i+p];
        if(edfhdr->plus_admincode[i]=='_')  edfhdr->plus_admincode[i] = ' ';
      }
      edfhdr->plus_admincode[i] = 0;
      p += i + 1;
    }

    if((edfhdr->recording[p]=='X') && (edfhdr->recording[p+1]==' '))
    {
      edfhdr->plus_technician[0] = 0;
      p += 2;
    }
    else
    {
      for(i=0; i<(80-p); i++)
      {
        if(edfhdr->recording[i+p]==' ')
        {
          break;
        }
        edfhdr->plus_technician[i] = edfhdr->recording[i+p];
        if(edfhdr->plus_technician[i]=='_')  edfhdr->plus_technician[i] = ' ';
      }
      edfhdr->plus_technician[i] = 0;
      p += i + 1;
    }

    if((edfhdr->recording[p]=='X') && (edfhdr->recording[p+1]==' '))
    {
      edfhdr->plus_equipment[0] = 0;
      p += 2;
    }
    else
    {
      for(i=0; i<(80-p); i++)
      {
        if(edfhdr->recording[i+p]==' ')
        {
          break;
        }
        edfhdr->plus_equipment[i] = edfhdr->recording[i+p];
        if(edfhdr->plus_equipment[i]=='_')  edfhdr->plus_equipment[i] = ' ';
      }
      edfhdr->plus_equipment[i] = 0;
      p += i + 1;
    }

    for(i=0; i<(80-p);i++)
    {
      edfhdr->plus_recording_additional[i] = edfhdr->recording[i+p];
    }
    edfhdr->plus_recording_additional[i] = 0;
    p += i + 1;
  }

/********************* FILESIZE *********************************************/

  edfhdr->hdrsize = edfhdr->edfsignals * 256 + 256;

  if (check_file_size != EDFLIB_DO_NOT_CHECK_FILE_SIZE)
  {  
	  fseeko(inputfile, 0LL, SEEK_END);
	   if(ftello(inputfile)<(edfhdr->recordsize * edfhdr->datarecords + edfhdr->hdrsize))
	  {
		printf("filesize %d != %d*%d+%d ",ftello(inputfile),edfhdr->recordsize, edfhdr->datarecords, edfhdr->hdrsize);
		*edf_error = EDFLIB_FILE_ERRORS_FILESIZE;
		free(edf_hdr);
		free(edfhdr->edfparam);
		free(edfhdr);
		return NULL;
	  }

  }
  n = 0;

  for(i=0; i<edfhdr->edfsignals; i++)
  {
    edfhdr->edfparam[i].buf_offset = n;
    if(edfhdr->bdf)  n += edfhdr->edfparam[i].smp_per_record * 3;
    else  n += edfhdr->edfparam[i].smp_per_record * 2;
	if ((edfhdr->edfparam[i].dig_max == edfhdr->edfparam[i].dig_min) || (edfhdr->edfparam[i].phys_max == edfhdr->edfparam[i].phys_min))
	{
		edfhdr->edfparam[i].bitvalue = 1;
		edfhdr->edfparam[i].offset = 0;
	}
	else
	{

    edfhdr->edfparam[i].bitvalue = (edfhdr->edfparam[i].phys_max - edfhdr->edfparam[i].phys_min) / (edfhdr->edfparam[i].dig_max - edfhdr->edfparam[i].dig_min);
    edfhdr->edfparam[i].offset = edfhdr->edfparam[i].phys_max / edfhdr->edfparam[i].bitvalue - edfhdr->edfparam[i].dig_max;
    }
  }

  edfhdr->file_hdl = inputfile;

  free(edf_hdr);

  return edfhdr;
}




static int edflib_is_integer_number(char *str)
{
  int i=0, l, hasspace = 0, hassign=0, digit=0;

  l = strlen(str);

  if(!l)  return 1;

  if((str[0]=='+')||(str[0]=='-'))
  {
    hassign++;
    i++;
  }

  for(; i<l; i++)
  {
    if(str[i]==' ')
    {
      if(!digit)
      {
        return 1;
      }
      hasspace++;
    }
    else
    {
      if((str[i]<48)||(str[i]>57))
      {
        return 1;
      }
      else
      {
        if(hasspace)
        {
          return 1;
        }
        digit++;
      }
    }
  }

  if(digit)  return 0;
  else  return 1;
}



static int edflib_is_number(char *str)
{
  int i=0, l, hasspace = 0, hassign=0, digit=0, hasdot=0, hasexp=0;

  l = strlen(str);

  if(!l)  return 1;

  if((str[0]=='+')||(str[0]=='-'))
  {
    hassign++;
    i++;
  }

  for(; i<l; i++)
  {
    if((str[i]=='e')||(str[i]=='E'))
    {
      if((!digit)||hasexp)
      {
        return 1;
      }
      hasexp++;
      hassign = 0;
      digit = 0;

      break;
    }

    if(str[i]==' ')
    {
      if(!digit)
      {
        return 1;
      }
      hasspace++;
    }
    else
    {
      if(((str[i]<48)||(str[i]>57))&&str[i]!='.')
      {
        return 1;
      }
      else
      {
        if(hasspace)
        {
          return 1;
        }
        if(str[i]=='.')
        {
          if(hasdot)  return 1;
          hasdot++;
        }
        else
        {
          digit++;
        }
      }
    }
  }

  if(hasexp)
  {
    if(++i==l)
    {
      return 1;
    }

    if((str[i]=='+')||(str[i]=='-'))
    {
      hassign++;
      i++;
    }

    for(; i<l; i++)
    {
      if(str[i]==' ')
      {
        if(!digit)
        {
          return 1;
        }
        hasspace++;
      }
      else
      {
        if((str[i]<48)||(str[i]>57))
        {
          return 1;
        }
        else
        {
          if(hasspace)
          {
            return 1;
          }

          digit++;
        }
      }
    }
  }

  if(digit)  return 0;
  else  return 1;
}


static long long edflib_get_long_duration(char *str)
{
  int i, len=8, hasdot=0, dotposition=0;
  long long value=0, radix;

  if((str[0] == '+') || (str[0] == '-'))
  {
    for(i=0; i<7; i++)
    {
      str[i] = str[i+1];
    }
    str[7] = ' ';
  }
  
  for(i=0; i<8; i++)
  {
    if(str[i]==' ')
    {
      len = i;
      break;
    }
  }

  for(i=0; i<len; i++)
  {
    if(str[i]=='.')
    {
      hasdot = 1;
      dotposition = i;
      break;
    }
  }

  if(hasdot)
  {
    radix = EDFLIB_TIME_DIMENSION;

    for(i=dotposition-1; i>=0; i--)
    {
        value += ((long long)(str[i] - 48)) * radix;
        radix *= 10;
    }

    radix = EDFLIB_TIME_DIMENSION / 10;

    for(i=dotposition+1; i<len; i++)
    {
        value += ((long long)(str[i] - 48)) * radix;
        radix /= 10;
    }
  }
  else
  {
    radix = EDFLIB_TIME_DIMENSION;

    for(i=len-1; i>=0; i--)
    {
        value += ((long long)(str[i] - 48)) * radix;
        radix *= 10;
    }
  }

  return value;
}


int edflib_version(void)
{
  return EDFLIB_VERSION;
}

int edflib_repair_file_size(const char *path, struct edfhdrblock *edfhdr)
{
  int p;
  FILE *file;

  file = fopeno(path, "wb");
  if(edfhdr->datarecords<100000000LL)
  {
    fseeko(file, 236LL, SEEK_SET);
    p = edflib_fprint_int_number_nonlocalized(file, (int)(edfhdr->datarecords), 0, 0);
    if(p < 2)
    {
      fputc(' ', file);
    }
  }
  fclose(file);
  return 0;
}

static int edflib_get_annotations(struct edfhdrblock *edfhdr, int hdl, int read_annotations)
{
  int i, j, k, p, r=0, n,
      edfsignals,
      datarecords,
      recordsize,
      discontinuous,
      *annot_ch,
      nr_annot_chns,
      max,
      onset,
      duration,
      duration_start,
      zero,
      max_tal_ln,
      error,
      annots_in_record,
      annots_in_tal,
      samplesize=2;

  char *scratchpad,
       *cnv_buf,
       *time_in_txt,
       *duration_in_txt;


  long long data_record_duration,
            elapsedtime,
            time_tmp=0;

  FILE *inputfile;

  struct edfparamblock *edfparam;

  struct edf_annotationblock *new_annotation=NULL,
                             *malloc_list;

  inputfile = edfhdr->file_hdl;
  edfsignals = edfhdr->edfsignals;
  recordsize = edfhdr->recordsize;
  edfparam = edfhdr->edfparam;
  nr_annot_chns = edfhdr->nr_annot_chns;
  datarecords = edfhdr->datarecords;
  data_record_duration = edfhdr->long_data_record_duration;
  discontinuous = edfhdr->discontinuous;
  annot_ch = edfhdr->annot_ch;

  if(edfhdr->edfplus)
  {
    samplesize = 2;
  }
  if(edfhdr->bdfplus)
  {
    samplesize = 3;
  }

  cnv_buf = (char *)calloc(1, recordsize);
  if(cnv_buf==NULL)
  {
    return 1;
  }

  max_tal_ln = 0;

  for(i=0; i<nr_annot_chns; i++)
  {
    if(max_tal_ln<edfparam[annot_ch[i]].smp_per_record * samplesize)  max_tal_ln = edfparam[annot_ch[i]].smp_per_record * samplesize;
  }

  if(max_tal_ln<128)  max_tal_ln = 128;

  scratchpad = (char *)calloc(1, max_tal_ln + 3);
  if(scratchpad==NULL)
  {
    free(cnv_buf);
    return 1;
  }

  time_in_txt = (char *)calloc(1, max_tal_ln + 3);
  if(time_in_txt==NULL)
  {
    free(cnv_buf);
    free(scratchpad);
    return 1;
  }

  duration_in_txt = (char *)calloc(1, max_tal_ln + 3);
  if(duration_in_txt==NULL)
  {
    free(cnv_buf);
    free(scratchpad);
    free(time_in_txt);
    return 1;
  }

  if(fseeko(inputfile, (long long)((edfsignals + 1) * 256), SEEK_SET))
  {
    free(cnv_buf);
    free(scratchpad);
    free(time_in_txt);
    free(duration_in_txt);
    return 2;
  }

  elapsedtime = 0;

  for(i=0; i<datarecords; i++)
  {
    if(fread(cnv_buf, recordsize, 1, inputfile)!=1)
    {
      free(cnv_buf);
      free(scratchpad);
      free(time_in_txt);
      free(duration_in_txt);
      return 2;
    }


/************** process annotationsignals (if any) **************/

    error = 0;

    for(r=0; r<nr_annot_chns; r++)
    {
      n = 0;
      zero = 0;
      onset = 0;
      duration = 0;
      duration_start = 0;
      scratchpad[0] = 0;
      annots_in_tal = 0;
      annots_in_record = 0;

      p = edfparam[annot_ch[r]].buf_offset;
      max = edfparam[annot_ch[r]].smp_per_record * samplesize;

/************** process one annotation signal ****************/

      if(cnv_buf[p + max - 1]!=0)
      {
        error = 5;
        goto END;
      }

      if(!r)  /* if it's the first annotation signal, then check */
      {       /* the timekeeping annotation */
        error = 1;

        for(k=0; k<(max-2); k++)
        {
          scratchpad[k] = cnv_buf[p + k];

          if(scratchpad[k]==20)
          {
            if(cnv_buf[p + k + 1]!=20)
            {
              error = 6;
              goto END;
            }
            scratchpad[k] = 0;
            if(edflib_is_onset_number(scratchpad))
            {
              error = 36;
              goto END;
            }
            else
            {
              time_tmp = edflib_get_long_time(scratchpad);
              if(i)
              {
                if(discontinuous)
                {
                  if((time_tmp-elapsedtime)<data_record_duration)
                  {
                    error = 4;
                    goto END;
                  }
                }
                else
                {
                  if((time_tmp-elapsedtime)!=data_record_duration)
                  {
                    error = 3;
                    goto END;
                  }
                }
              }
              else
              {
                if(time_tmp>=EDFLIB_TIME_DIMENSION)
                {
                  error = 2;
                  goto END;
                }
                else
                {
                  edfhdr->starttime_offset = time_tmp;
                }
              }
              elapsedtime = time_tmp;
              error = 0;
              break;
            }
          }
        }
      }

      for(k=0; k<max; k++)
      {
        scratchpad[n] = cnv_buf[p + k];

        if(!scratchpad[n])
        {
          if(!zero)
          {
            if(k)
            {
              if(cnv_buf[p + k - 1]!=20)
              {
                error = 33;
                goto END;
              }
            }
            n = 0;
            onset = 0;
            duration = 0;
            duration_start = 0;
            scratchpad[0] = 0;
            annots_in_tal = 0;
          }
          zero++;
          continue;
        }
        if(zero>1)
        {
          error = 34;
          goto END;
        }
        zero = 0;

        if((scratchpad[n]==20)||(scratchpad[n]==21))
        {
          if(scratchpad[n]==21)
          {
            if(duration||duration_start||onset||annots_in_tal)
            {               /* it's not allowed to have multiple duration fields */
              error = 35;   /* in one TAL or to have a duration field which is   */
              goto END;     /* not immediately behind the onsetfield             */
            }
            duration_start = 1;
          }

          if((scratchpad[n]==20)&&onset&&(!duration_start))
          {
            if(r||annots_in_record)
            {
              if(n >= 0)
              {
                if(edfhdr->annots_in_file >= edfhdr->annotlist_sz)
                {
                  malloc_list = (struct edf_annotationblock *)realloc(annotationslist[hdl],
                                                                      sizeof(struct edf_annotationblock) * (edfhdr->annotlist_sz + EDFLIB_ANNOT_MEMBLOCKSZ));
                  if(malloc_list==NULL)
                  {
                    free(cnv_buf);
                    free(scratchpad);
                    free(time_in_txt);
                    free(duration_in_txt);
                    return -1;
                  }

                  annotationslist[hdl] = malloc_list;

                  edfhdr->annotlist_sz += EDFLIB_ANNOT_MEMBLOCKSZ;
                }

                new_annotation = annotationslist[hdl] + edfhdr->annots_in_file;

                new_annotation->annotation[0] = 0;

                if(duration)  edflib_strlcpy(new_annotation->duration, duration_in_txt, 16);
                else  new_annotation->duration[0] = 0;

                for(j=0; j<n; j++)
                {
                  if(j==EDFLIB_MAX_ANNOTATION_LEN)  break;
                  new_annotation->annotation[j] = scratchpad[j];
                }
                new_annotation->annotation[j] = 0;

                new_annotation->onset = edflib_get_long_time(time_in_txt);

                new_annotation->onset -= edfhdr->starttime_offset;

                edfhdr->annots_in_file++;

                if(read_annotations==EDFLIB_READ_ANNOTATIONS)
                {
                  if(!(strncmp(new_annotation->annotation, "Recording ends", 14)))
                  {
                    if(nr_annot_chns==1)
                    {
                      goto END;
                    }
                  }
                }
              }
            }

            annots_in_tal++;
            annots_in_record++;
            n = 0;
            continue;
          }

          if(!onset)
          {
            scratchpad[n] = 0;
            if(edflib_is_onset_number(scratchpad))
            {
              error = 36;
              goto END;
            }
            onset = 1;
            n = 0;
            edflib_strlcpy(time_in_txt, scratchpad, max_tal_ln + 3);
            continue;
          }

          if(duration_start)
          {
            scratchpad[n] = 0;
            if(edflib_is_duration_number(scratchpad))
            {
              error = 37;
              goto END;
            }

            for(j=0; j<n; j++)
            {
              if(j==15)  break;
              duration_in_txt[j] = scratchpad[j];
              if((duration_in_txt[j]<32)||(duration_in_txt[j]>126))
              {
                duration_in_txt[j] = '.';
              }
            }
            duration_in_txt[j] = 0;

            duration = 1;
            duration_start = 0;
            n = 0;
            continue;
          }
        }

        n++;
      }

 END:

/****************** end ************************/

      if(error)
      {
        free(cnv_buf);
        free(scratchpad);
        free(time_in_txt);
        free(duration_in_txt);
        return 9;
      }
    }
  }

  free(cnv_buf);
  free(scratchpad);
  free(time_in_txt);
  free(duration_in_txt);

  return 0;
}


static int edflib_is_duration_number(char *str)
{
  int i, l, hasdot = 0;

  l = strlen(str);

  if(!l)  return 1;

  if((str[0] == '.')||(str[l-1] == '.'))  return 1;

  for(i=0; i<l; i++)
  {
    if(str[i]=='.')
    {
      if(hasdot)  return 1;
      hasdot++;
    }
    else
    {
      if((str[i]<48)||(str[i]>57))  return 1;
    }
  }

  return 0;
}



static int edflib_is_onset_number(char *str)
{
  int i, l, hasdot = 0;

  l = strlen(str);

  if(l<2)  return 1;

  if((str[0]!='+')&&(str[0]!='-'))  return 1;

  if((str[1] == '.')||(str[l-1] == '.'))  return 1;

  for(i=1; i<l; i++)
  {
    if(str[i]=='.')
    {
      if(hasdot)  return 1;
      hasdot++;
    }
    else
    {
      if((str[i]<48)||(str[i]>57))  return 1;
    }
  }

  return 0;
}



static long long edflib_get_long_time(char *str)
{
  int i, len, hasdot=0, dotposition=0, neg=0;

  long long value=0, radix;

  if(str[0] == '+')
  {
    str++;
  }
  else if(str[0] == '-')
    {
      neg = 1;
      str++;
    }

  len = strlen(str);

  for(i=0; i<len; i++)
  {
    if(str[i]=='.')
    {
      hasdot = 1;
      dotposition = i;
      break;
    }
  }

  if(hasdot)
  {
    radix = EDFLIB_TIME_DIMENSION;

    for(i=dotposition-1; i>=0; i--)
    {
        value += ((long long)(str[i] - 48)) * radix;
        radix *= 10;
    }

    radix = EDFLIB_TIME_DIMENSION / 10;

    for(i=dotposition+1; i<len; i++)
    {
        value += ((long long)(str[i] - 48)) * radix;
        radix /= 10;
    }
  }
  else
  {
    radix = EDFLIB_TIME_DIMENSION;

    for(i=len-1; i>=0; i--)
    {
        value += ((long long)(str[i] - 48)) * radix;
        radix *= 10;
    }
  }

  if(neg)  value = -value;

  return value;
}


static void edflib_latin1_to_ascii(char *str, int len)
{
  int i, value;

  for(i=0; i<len; i++)
  {
    value = *((unsigned char *)(str + i));

    if((value>31)&&(value<127))
    {
      continue;
    }

    switch(value)
    {
      case 128 : str[i] = 'E';  break;

      case 130 : str[i] = ',';  break;

      case 131 : str[i] = 'F';  break;

      case 132 : str[i] = '\"';  break;

      case 133 : str[i] = '.';  break;

      case 134 : str[i] = '+';  break;

      case 135 : str[i] = '+';  break;

      case 136 : str[i] = '^';  break;

      case 137 : str[i] = 'm';  break;

      case 138 : str[i] = 'S';  break;

      case 139 : str[i] = '<';  break;

      case 140 : str[i] = 'E';  break;

      case 142 : str[i] = 'Z';  break;

      case 145 : str[i] = '`';  break;

      case 146 : str[i] = '\'';  break;

      case 147 : str[i] = '\"';  break;

      case 148 : str[i] = '\"';  break;

      case 149 : str[i] = '.';  break;

      case 150 : str[i] = '-';  break;

      case 151 : str[i] = '-';  break;

      case 152 : str[i] = '~';  break;

      case 154 : str[i] = 's';  break;

      case 155 : str[i] = '>';  break;

      case 156 : str[i] = 'e';  break;

      case 158 : str[i] = 'z';  break;

      case 159 : str[i] = 'Y';  break;

      case 171 : str[i] = '<';  break;

      case 180 : str[i] = '\'';  break;

      case 181 : str[i] = 'u';  break;

      case 187 : str[i] = '>';  break;

      case 191 : str[i] = '\?';  break;

      case 192 : str[i] = 'A';  break;

      case 193 : str[i] = 'A';  break;

      case 194 : str[i] = 'A';  break;

      case 195 : str[i] = 'A';  break;

      case 196 : str[i] = 'A';  break;

      case 197 : str[i] = 'A';  break;

      case 198 : str[i] = 'E';  break;

      case 199 : str[i] = 'C';  break;

      case 200 : str[i] = 'E';  break;

      case 201 : str[i] = 'E';  break;

      case 202 : str[i] = 'E';  break;

      case 203 : str[i] = 'E';  break;

      case 204 : str[i] = 'I';  break;

      case 205 : str[i] = 'I';  break;

      case 206 : str[i] = 'I';  break;

      case 207 : str[i] = 'I';  break;

      case 208 : str[i] = 'D';  break;

      case 209 : str[i] = 'N';  break;

      case 210 : str[i] = 'O';  break;

      case 211 : str[i] = 'O';  break;

      case 212 : str[i] = 'O';  break;

      case 213 : str[i] = 'O';  break;

      case 214 : str[i] = 'O';  break;

      case 215 : str[i] = 'x';  break;

      case 216 : str[i] = 'O';  break;

      case 217 : str[i] = 'U';  break;

      case 218 : str[i] = 'U';  break;

      case 219 : str[i] = 'U';  break;

      case 220 : str[i] = 'U';  break;

      case 221 : str[i] = 'Y';  break;

      case 222 : str[i] = 'I';  break;

      case 223 : str[i] = 's';  break;

      case 224 : str[i] = 'a';  break;

      case 225 : str[i] = 'a';  break;

      case 226 : str[i] = 'a';  break;

      case 227 : str[i] = 'a';  break;

      case 228 : str[i] = 'a';  break;

      case 229 : str[i] = 'a';  break;

      case 230 : str[i] = 'e';  break;

      case 231 : str[i] = 'c';  break;

      case 232 : str[i] = 'e';  break;

      case 233 : str[i] = 'e';  break;

      case 234 : str[i] = 'e';  break;

      case 235 : str[i] = 'e';  break;

      case 236 : str[i] = 'i';  break;

      case 237 : str[i] = 'i';  break;

      case 238 : str[i] = 'i';  break;

      case 239 : str[i] = 'i';  break;

      case 240 : str[i] = 'd';  break;

      case 241 : str[i] = 'n';  break;

      case 242 : str[i] = 'o';  break;

      case 243 : str[i] = 'o';  break;

      case 244 : str[i] = 'o';  break;

      case 245 : str[i] = 'o';  break;

      case 246 : str[i] = 'o';  break;

      case 247 : str[i] = '-';  break;

      case 248 : str[i] = '0';  break;

      case 249 : str[i] = 'u';  break;

      case 250 : str[i] = 'u';  break;

      case 251 : str[i] = 'u';  break;

      case 252 : str[i] = 'u';  break;

      case 253 : str[i] = 'y';  break;

      case 254 : str[i] = 't';  break;

      case 255 : str[i] = 'y';  break;

      default  : str[i] = ' ';  break;
    }
  }
}


static void edflib_latin12utf8(char *latin1_str, int len)
{
  int i, j;

  unsigned char *str, tmp_str[512];


  str = (unsigned char *)latin1_str;

  j = 0;

  for(i=0; i<len; i++)
  {
    if(str[i]==0)
    {
      tmp_str[j] = 0;

      break;
    }

    tmp_str[j] = str[i];

    if(str[i]<32) tmp_str[j] = '.';

    if((str[i]>126)&&(str[i]<160))  tmp_str[j] = '.';

    if(str[i]>159)
    {
      if((len-j)<2)
      {
        tmp_str[j] = ' ';
      }
      else
      {
        tmp_str[j] = 192 + (str[i]>>6);
        j++;
        tmp_str[j] = 128 + (str[i]&63);
      }
    }

    j++;

    if(j>=len)  break;
  }

  for(i=0; i<len; i++)
  {
    str[i] = tmp_str[i];
  }
}


int edfopen_file_writeonly(const char *path, int filetype, int number_of_signals)
{
  int i, handle;

  FILE *file;

  struct edfhdrblock *hdr;

  /*
  if((filetype!=EDFLIB_FILETYPE_EDFPLUS)&&(filetype!=EDFLIB_FILETYPE_BDFPLUS))
  {
    return EDFLIB_FILETYPE_ERROR;
  }
  */


  if(edf_files_open>=EDFLIB_MAXFILES)
  {
    return EDFLIB_MAXFILES_REACHED;
  }

  for(i=0; i<EDFLIB_MAXFILES; i++)
  {
    if(hdrlist[i]!=NULL)
    {
      if(!(strcmp(path, hdrlist[i]->path)))
      {
        return EDFLIB_FILE_ALREADY_OPENED;
      }
    }
  }

  if(number_of_signals<0)
  {
    return EDFLIB_NUMBER_OF_SIGNALS_INVALID;
  }

  if(number_of_signals>EDFLIB_MAXSIGNALS)
  {
    return EDFLIB_NUMBER_OF_SIGNALS_INVALID;
  }

  hdr = (struct edfhdrblock *)calloc(1, sizeof(struct edfhdrblock));
  if(hdr==NULL)
  {
    return EDFLIB_MALLOC_ERROR;
  }

  hdr->edfparam = (struct edfparamblock *)calloc(1, sizeof(struct edfparamblock) * number_of_signals);
  if(hdr->edfparam==NULL)
  {
    free(hdr);

    return EDFLIB_MALLOC_ERROR;
  }

  hdr->writemode = 1;

  hdr->edfsignals = number_of_signals;

  handle = -1;

  for(i=0; i<EDFLIB_MAXFILES; i++)
  {
    if(hdrlist[i]==NULL)
    {
      hdrlist[i] = hdr;

      handle = i;

      break;
    }
  }

  if(handle<0)
  {
    free(hdr->edfparam);

    free(hdr);

    return EDFLIB_MAXFILES_REACHED;
  }

  write_annotationslist[handle] = NULL;

  hdr->annotlist_sz = 0;

  hdr->annots_in_file = 0;

  file = fopeno(path, "wb");
  if(file==NULL)
  {
    free(hdr->edfparam);
    hdr->edfparam = NULL;
    free(hdr);
    hdr = NULL;
    hdrlist[handle] = NULL;

    return EDFLIB_NO_SUCH_FILE_OR_DIRECTORY;
  }

  hdr->file_hdl = file;

  edflib_strlcpy(hdr->path, path, 1024);

  edf_files_open++;

  if(filetype==EDFLIB_FILETYPE_EDFPLUS)
  {
    hdr->edf = 1;
    hdr->edfplus = 1;
	hdr->nr_annot_chns = 1;	
	
  }
  
  if(filetype==EDFLIB_FILETYPE_EDF)
  {
    hdr->edf = 1;
    hdr->edfplus = 0;
    hdr->nr_annot_chns = 0;
  }
  
  if(filetype==EDFLIB_FILETYPE_BDFPLUS)
  {
    hdr->bdf = 1;
    hdr->bdfplus = 1;
	hdr->nr_annot_chns = 1;	
  }
  
  if(filetype==EDFLIB_FILETYPE_BDF)
  {
    hdr->bdf = 1;
    hdr->bdfplus = 0;
    hdr->nr_annot_chns = 0;
  }

  hdr->long_data_record_duration = EDFLIB_TIME_DIMENSION;

  hdr->data_record_duration = 1.0;

  return handle;
}


int edf_set_samplefrequency(int handle, int edfsignal, int samplefrequency)
{
  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  if(!(hdrlist[handle]->writemode))
  {
    return -1;
  }

  if(edfsignal<0)
  {
    return -1;
  }

  if(edfsignal>=hdrlist[handle]->edfsignals)
  {
    return -1;
  }

  if(samplefrequency<1)
  {
    return -1;
  }

  if(hdrlist[handle]->datarecords)
  {
    return -1;
  }

  hdrlist[handle]->edfparam[edfsignal].smp_per_record = samplefrequency;

  return 0;
}


int edf_set_number_of_annotation_signals(int handle, int annot_signals)
{
  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  if(!(hdrlist[handle]->writemode))
  {
    return -1;
  }

  if(hdrlist[handle]->datarecords)
  {
    return -1;
  }

  if((annot_signals < 1) || (annot_signals > EDFLIB_MAX_ANNOTATION_CHANNELS))
  {
    return -1;
  }

  hdrlist[handle]->nr_annot_chns = annot_signals;

  return 0;
}


int edf_set_datarecord_duration(int handle, int duration)
{
  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  if(!(hdrlist[handle]->writemode))
  {
    return -1;
  }

  if(hdrlist[handle]->datarecords)
  {
    return -1;
  }

  if((duration < 100) || (duration > 6000000))
  {
    return -1;
  }

  hdrlist[handle]->long_data_record_duration = (long long)duration * 100LL;

  if(hdrlist[handle]->long_data_record_duration < (EDFLIB_TIME_DIMENSION * 10LL))
  {
    hdrlist[handle]->long_data_record_duration /= 10LL;

    hdrlist[handle]->long_data_record_duration *= 10LL;
  }
  else
  {
    hdrlist[handle]->long_data_record_duration /= 100LL;

    hdrlist[handle]->long_data_record_duration *= 100LL;
  }

  hdrlist[handle]->data_record_duration = ((double)(hdrlist[handle]->long_data_record_duration)) / EDFLIB_TIME_DIMENSION;

  return 0;
}


int edf_set_micro_datarecord_duration(int handle, int duration)
{
  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  if(!(hdrlist[handle]->writemode))
  {
    return -1;
  }

  if(hdrlist[handle]->datarecords)
  {
    return -1;
  }

  if((duration < 1) || (duration > 9999))
  {
    return -1;
  }

  hdrlist[handle]->long_data_record_duration = (long long)duration * 10LL;

  hdrlist[handle]->data_record_duration = ((double)(hdrlist[handle]->long_data_record_duration)) / EDFLIB_TIME_DIMENSION;

  return 0;
}


int edf_set_subsecond_starttime(int handle, int subsecond)
{
  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  if(!(hdrlist[handle]->writemode))
  {
    return -1;
  }

  if(hdrlist[handle]->datarecords)
  {
    return -1;
  }

  if((subsecond < 0) || (subsecond > 9999999))
  {
    return -1;
  }

  hdrlist[handle]->starttime_offset = (long long)subsecond;

  return 0;
}


int edfwrite_digital_short_samples(int handle, short *buf)
{
  int  i,
       error,
       sf,
       digmax,
       digmin,
       edfsignal,
       value;

  FILE *file;

  struct edfhdrblock *hdr;


  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  if(!(hdrlist[handle]->writemode))
  {
    return -1;
  }

  if(hdrlist[handle]->edfsignals == 0)
  {
    return -1;
  }

  if(hdrlist[handle]->bdf == 1)
  {
    return -1;
  }

  hdr = hdrlist[handle];

  file = hdr->file_hdl;

  edfsignal = hdr->signal_write_sequence_pos;

  if(!hdr->datarecords)
  {
    if(!edfsignal)
    {
      error = edflib_write_edf_header(hdr);

      if(error)
      {
        return error;
      }
    }
  }

  sf = hdr->edfparam[edfsignal].smp_per_record;

  digmax = hdr->edfparam[edfsignal].dig_max;

  digmin = hdr->edfparam[edfsignal].dig_min;

  if(hdr->edf)
  {
    if((digmax != 32767) || (digmin != -32768))
    {
      for(i=0; i<sf; i++)
      {
        if(buf[i]>digmax)
        {
          buf[i] = digmax;
        }

        if(buf[i]<digmin)
        {
          buf[i] = digmin;
        }
      }
    }

    if(fwrite(buf, sf * 2, 1, file) != 1)
    {
      return -1;
    }
  }
  else  // BDF
  {
    if(hdr->wrbufsize < (sf * 3))
    {
      free(hdr->wrbuf);

      hdr->wrbufsize = 0;

      hdr->wrbuf = (char *)malloc(sf * 3);

      if(hdr->wrbuf == NULL)
      {
        return -1;
      }

      hdr->wrbufsize = sf * 3;

    }

    for(i=0; i<sf; i++)
    {
      value = buf[i];

      if(value>digmax)
      {
        value = digmax;
      }

      if(value<digmin)
      {
        value = digmin;
      }

      hdr->wrbuf[i * 3] = value & 0xff;

      hdr->wrbuf[i * 3 + 1] = (value >> 8) & 0xff;

      hdr->wrbuf[i * 3 + 2] = (value >> 16) & 0xff;
    }

    if(fwrite(hdr->wrbuf, sf * 3, 1, file) != 1)
    {
      return -1;
    }
  }

  hdr->signal_write_sequence_pos++;

  if(hdr->signal_write_sequence_pos == hdr->edfsignals)
  {
    hdr->signal_write_sequence_pos = 0;

    if(edflib_write_tal(hdr, file))
    {
      return -1;
    }

    hdr->datarecords++;

    fflush(file);
  }

  return 0;
}


int edfwrite_digital_samples(int handle, int *buf)
{
  int  i,
       error,
       sf,
       digmax,
       digmin,
       edfsignal,
       value;

  FILE *file;

  struct edfhdrblock *hdr;


  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  if(!(hdrlist[handle]->writemode))
  {
    return -1;
  }

  if(hdrlist[handle]->edfsignals == 0)
  {
    return -1;
  }

  hdr = hdrlist[handle];

  file = hdr->file_hdl;

  edfsignal = hdr->signal_write_sequence_pos;

  if(!hdr->datarecords)
  {
    if(!edfsignal)
    {
      error = edflib_write_edf_header(hdr);

      if(error)
      {
        return error;
      }
    }
  }

  sf = hdr->edfparam[edfsignal].smp_per_record;

  digmax = hdr->edfparam[edfsignal].dig_max;

  digmin = hdr->edfparam[edfsignal].dig_min;

  if(hdr->edf)
  {
    if(hdr->wrbufsize < (sf * 2))
    {
      free(hdr->wrbuf);

      hdr->wrbufsize = 0;

      hdr->wrbuf = (char *)malloc(sf * 2);

      if(hdr->wrbuf == NULL)
      {
        return -1;
      }

      hdr->wrbufsize = sf * 2;
    }

    for(i=0; i<sf; i++)
    {
      value = buf[i];

      if(value>digmax)
      {
        value = digmax;
      }

      if(value<digmin)
      {
        value = digmin;
      }

      hdr->wrbuf[i * 2] = value & 0xff;

      hdr->wrbuf[i * 2 + 1] = (value >> 8) & 0xff;
    }

    if(fwrite(hdr->wrbuf, sf * 2, 1, file) != 1)
    {
      return -1;
    }
  }
  else  // BDF
  {
    if(hdr->wrbufsize < (sf * 3))
    {
      free(hdr->wrbuf);

      hdr->wrbufsize = 0;

      hdr->wrbuf = (char *)malloc(sf * 3);

      if(hdr->wrbuf == NULL)
      {
        return -1;
      }

      hdr->wrbufsize = sf * 3;
    }

    for(i=0; i<sf; i++)
    {
      value = buf[i];

      if(value>digmax)
      {
        value = digmax;
      }

      if(value<digmin)
      {
        value = digmin;
      }

      hdr->wrbuf[i * 3] = value & 0xff;

      hdr->wrbuf[i * 3 + 1] = (value >> 8) & 0xff;

      hdr->wrbuf[i * 3 + 2] = (value >> 16) & 0xff;

    }

    if(fwrite(hdr->wrbuf, sf * 3, 1, file) != 1)
    {
      return -1;
    }
  }

  hdr->signal_write_sequence_pos++;

  if(hdr->signal_write_sequence_pos == hdr->edfsignals)
  {
    hdr->signal_write_sequence_pos = 0;

    if(edflib_write_tal(hdr, file))
    {
      return -1;
    }

    hdr->datarecords++;

    fflush(file);
  }

  return 0;
}


int edf_blockwrite_digital_samples(int handle, int *buf)
{
  int  i, j,
       error,
       sf,
       digmax,
       digmin,
       edfsignals,
       buf_offset,
       value;

  FILE *file;

  struct edfhdrblock *hdr;


  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  if(!(hdrlist[handle]->writemode))
  {
    return -1;
  }

  if(hdrlist[handle]->signal_write_sequence_pos)
  {
    return -1;
  }

  if(hdrlist[handle]->edfsignals == 0)
  {
    return -1;
  }

  hdr = hdrlist[handle];

  file = hdr->file_hdl;

  edfsignals = hdr->edfsignals;

  if(!hdr->datarecords)
  {
    error = edflib_write_edf_header(hdr);

    if(error)
    {
      return error;
    }
  }

  buf_offset = 0;

  for(j=0; j<edfsignals; j++)
  {
    sf = hdr->edfparam[j].smp_per_record;

    digmax = hdr->edfparam[j].dig_max;

    digmin = hdr->edfparam[j].dig_min;

    if(hdr->edf)
    {
      if(hdr->wrbufsize < (sf * 2))
      {
        free(hdr->wrbuf);

        hdr->wrbufsize = 0;

        hdr->wrbuf = (char *)malloc(sf * 2);

        if(hdr->wrbuf == NULL)
        {
          return -1;
        }

        hdr->wrbufsize = sf * 2;
      }

      for(i=0; i<sf; i++)
      {
        value = buf[i + buf_offset];

        if(value>digmax)
        {
          value = digmax;
        }

        if(value<digmin)
        {
          value = digmin;
        }

        hdr->wrbuf[i * 2] = value & 0xff;

        hdr->wrbuf[i * 2 + 1] = (value >> 8) & 0xff;
      }

      if(fwrite(hdr->wrbuf, sf * 2, 1, file) != 1)
      {
        return -1;
      }
    }
    else  // BDF
    {
      if(hdr->wrbufsize < (sf * 3))
      {
        free(hdr->wrbuf);

        hdr->wrbufsize = 0;

        hdr->wrbuf = (char *)malloc(sf * 3);

        if(hdr->wrbuf == NULL)
        {
          return -1;
        }

        hdr->wrbufsize = sf * 3;
      }

      for(i=0; i<sf; i++)
      {
        value = buf[i + buf_offset];

        if(value>digmax)
        {
          value = digmax;
        }

        if(value<digmin)
        {
          value = digmin;
        }

        hdr->wrbuf[i * 3] = value & 0xff;

        hdr->wrbuf[i * 3 + 1] = (value >> 8) & 0xff;

        hdr->wrbuf[i * 3 + 2] = (value >> 16) & 0xff;
      }

      if(fwrite(hdr->wrbuf, sf * 3, 1, file) != 1)
      {
        return -1;
      }
    }

    buf_offset += sf;
  }

  if(edflib_write_tal(hdr, file))
  {
    return -1;
  }

  hdr->datarecords++;

  fflush(file);

  return 0;
}


int edf_blockwrite_digital_short_samples(int handle, short *buf)
{
  int  i, j,
       error,
       sf,
       digmax,
       digmin,
       edfsignals,
       buf_offset,
       value;

  FILE *file;

  struct edfhdrblock *hdr;


  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  if(!(hdrlist[handle]->writemode))
  {
    return -1;
  }

  if(hdrlist[handle]->signal_write_sequence_pos)
  {
    return -1;
  }

  if(hdrlist[handle]->edfsignals == 0)
  {
    return -1;
  }

  if(hdrlist[handle]->bdf == 1)
  {
    return -1;
  }

  hdr = hdrlist[handle];

  file = hdr->file_hdl;

  edfsignals = hdr->edfsignals;

  if(!hdr->datarecords)
  {
    error = edflib_write_edf_header(hdr);

    if(error)
    {
      return error;
    }
  }

  buf_offset = 0;

  for(j=0; j<edfsignals; j++)
  {
    sf = hdr->edfparam[j].smp_per_record;

    digmax = hdr->edfparam[j].dig_max;

    digmin = hdr->edfparam[j].dig_min;

    if(hdr->edf)
    {
      if((digmax != 32767) || (digmin != -32768))
      {
        for(i=0; i<sf; i++)
        {
          if(buf[i + buf_offset] > digmax)
          {
            buf[i + buf_offset] = digmax;
          }

          if(buf[i + buf_offset] < digmin)
          {
            buf[i + buf_offset] = digmin;
          }
        }
      }

      if(fwrite(buf + buf_offset, sf * 2, 1, file) != 1)
      {
        return -1;
      }
    }
    else  // BDF
    {
      if(hdr->wrbufsize < (sf * 3))
      {
        free(hdr->wrbuf);

        hdr->wrbufsize = 0;

        hdr->wrbuf = (char *)malloc(sf * 3);

        if(hdr->wrbuf == NULL)
        {
          return -1;
        }

        hdr->wrbufsize = sf * 3;
      }

      for(i=0; i<sf; i++)
      {
        value = buf[i + buf_offset];

        if(value>digmax)
        {
          value = digmax;
        }

        if(value<digmin)
        {
          value = digmin;
        }

        hdr->wrbuf[i * 3] = value & 0xff;

        hdr->wrbuf[i * 3 + 1] = (value >> 8) & 0xff;

        hdr->wrbuf[i * 3 + 2] = (value >> 16) & 0xff;
      }

      if(fwrite(hdr->wrbuf, sf * 3, 1, file) != 1)
      {
        return -1;
      }
    }

    buf_offset += sf;
  }

  if(edflib_write_tal(hdr, file))
  {
    return -1;
  }

  hdr->datarecords++;

  fflush(file);

  return 0;
}


int edf_blockwrite_digital_3byte_samples(int handle, void *buf)
{
  int  j,
       error,
       edfsignals,
       total_samples=0;

  FILE *file;

  struct edfhdrblock *hdr;


  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  if(!(hdrlist[handle]->writemode))
  {
    return -1;
  }

  if(hdrlist[handle]->signal_write_sequence_pos)
  {
    return -1;
  }

  if(hdrlist[handle]->edfsignals == 0)
  {
    return -1;
  }

  if(hdrlist[handle]->bdf != 1)
  {
    return -1;
  }

  hdr = hdrlist[handle];

  file = hdr->file_hdl;

  edfsignals = hdr->edfsignals;

  if(!hdr->datarecords)
  {
    error = edflib_write_edf_header(hdr);

    if(error)
    {
      return error;
    }
  }

  for(j=0; j<edfsignals; j++)
  {
    total_samples += hdr->edfparam[j].smp_per_record;
  }

  if(fwrite(buf, total_samples * 3, 1, file) != 1)
  {
    return -1;
  }

  if(edflib_write_tal(hdr, file))
  {
    return -1;
  }

  hdr->datarecords++;

  fflush(file);

  return 0;
}


int edfwrite_physical_samples(int handle, double *buf)
{
  int  i,
       error,
       sf,
       digmax,
       digmin,
       value,
       edfsignal;

  double bitvalue,
         phys_offset;

  FILE *file;

  struct edfhdrblock *hdr;


  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  if(!(hdrlist[handle]->writemode))
  {
    return -1;
  }

  if(hdrlist[handle]->edfsignals == 0)
  {
    return -1;
  }

  hdr = hdrlist[handle];

  file = hdr->file_hdl;

  edfsignal = hdr->signal_write_sequence_pos;

  if(!hdr->datarecords)
  {
    if(!edfsignal)
    {
      error = edflib_write_edf_header(hdr);

      if(error)
      {
        return error;
      }
    }
  }

  sf = hdr->edfparam[edfsignal].smp_per_record;

  digmax = hdr->edfparam[edfsignal].dig_max;

  digmin = hdr->edfparam[edfsignal].dig_min;

  bitvalue = hdr->edfparam[edfsignal].bitvalue;

  phys_offset = hdr->edfparam[edfsignal].offset;

  if(hdr->edf)
  {
    if(hdr->wrbufsize < (sf * 2))
    {
      free(hdr->wrbuf);

      hdr->wrbufsize = 0;

      hdr->wrbuf = (char *)malloc(sf * 2);

      if(hdr->wrbuf == NULL)
      {
        return -1;
      }

      hdr->wrbufsize = sf * 2;
    }

    for(i=0; i<sf; i++)
    {
      value = (buf[i] / bitvalue) - phys_offset;

      if(value>digmax)
      {
        value = digmax;
      }

      if(value<digmin)
      {
        value = digmin;
      }

      hdr->wrbuf[i * 2] = value & 0xff;

      hdr->wrbuf[i * 2 + 1] = (value >> 8) & 0xff;
    }

    if(fwrite(hdr->wrbuf, sf * 2, 1, file) != 1)
    {
      return -1;
    }
  }
  else  // BDF
  {
    if(hdr->wrbufsize < (sf * 3))
    {
      free(hdr->wrbuf);

      hdr->wrbufsize = 0;

      hdr->wrbuf = (char *)malloc(sf * 3);

      if(hdr->wrbuf == NULL)
      {
        return -1;
      }

      hdr->wrbufsize = sf * 3;
    }

    for(i=0; i<sf; i++)
    {
      value = (buf[i] / bitvalue) - phys_offset;

      if(value>digmax)
      {
        value = digmax;
      }

      if(value<digmin)
      {
        value = digmin;
      }

      hdr->wrbuf[i * 3] = value & 0xff;

      hdr->wrbuf[i * 3 + 1] = (value >> 8) & 0xff;

      hdr->wrbuf[i * 3 + 2] = (value >> 16) & 0xff;
    }

    if(fwrite(hdr->wrbuf, sf * 3, 1, file) != 1)
    {
      return -1;
    }
  }

  hdr->signal_write_sequence_pos++;

  if(hdr->signal_write_sequence_pos == hdr->edfsignals)
  {
    hdr->signal_write_sequence_pos = 0;

    if(edflib_write_tal(hdr, file))
    {
      return -1;
    }

    hdr->datarecords++;

    fflush(file);
  }

  return 0;
}


int edf_blockwrite_physical_samples(int handle, double *buf)
{
  int  i, j,
       error,
       sf,
       digmax,
       digmin,
       edfsignals,
       buf_offset,
       value;

  double bitvalue,
         phys_offset;

  FILE *file;

  struct edfhdrblock *hdr;


  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  if(!(hdrlist[handle]->writemode))
  {
    return -1;
  }

  if(hdrlist[handle]->signal_write_sequence_pos)
  {
    return -1;
  }

  if(hdrlist[handle]->edfsignals == 0)
  {
    return -1;
  }

  hdr = hdrlist[handle];

  file = hdr->file_hdl;

  edfsignals = hdr->edfsignals;

  if(!hdr->datarecords)
  {
    error = edflib_write_edf_header(hdr);

    if(error)
    {
      return error;
    }
  }

  buf_offset = 0;

  for(j=0; j<edfsignals; j++)
  {
    sf = hdr->edfparam[j].smp_per_record;

    digmax = hdr->edfparam[j].dig_max;

    digmin = hdr->edfparam[j].dig_min;

    bitvalue = hdr->edfparam[j].bitvalue;

    phys_offset = hdr->edfparam[j].offset;

    if(hdr->edf)
    {
      if(hdr->wrbufsize < (sf * 2))
      {
        free(hdr->wrbuf);

        hdr->wrbufsize = 0;

        hdr->wrbuf = (char *)malloc(sf * 2);

        if(hdr->wrbuf == NULL)
        {
          return -1;
        }

        hdr->wrbufsize = sf * 2;
      }

      for(i=0; i<sf; i++)
      {
        value = (buf[i + buf_offset] / bitvalue) - phys_offset;

        if(value>digmax)
        {
          value = digmax;
        }

        if(value<digmin)
        {
          value = digmin;
        }

        hdr->wrbuf[i * 2] = value & 0xff;

        hdr->wrbuf[i * 2 + 1] = (value >> 8) & 0xff;
      }

      if(fwrite(hdr->wrbuf, sf * 2, 1, file) != 1)
      {
        return -1;
      }
    }
    else  // BDF
    {
      if(hdr->wrbufsize < (sf * 3))
      {
        free(hdr->wrbuf);

        hdr->wrbufsize = 0;

        hdr->wrbuf = (char *)malloc(sf * 3);

        if(hdr->wrbuf == NULL)
        {
          return -1;
        }

        hdr->wrbufsize = sf * 3;
      }

      for(i=0; i<sf; i++)
      {
        value = (buf[i + buf_offset] / bitvalue) - phys_offset;

        if(value>digmax)
        {
          value = digmax;
        }

        if(value<digmin)
        {
          value = digmin;
        }

        hdr->wrbuf[i * 3] = value & 0xff;

        hdr->wrbuf[i * 3 + 1] = (value >> 8) & 0xff;

        hdr->wrbuf[i * 3 + 2] = (value >> 16) & 0xff;
      }

      if(fwrite(hdr->wrbuf, sf * 3, 1, file) != 1)
      {
        return -1;
      }
    }

    buf_offset += sf;
  }

  if(edflib_write_tal(hdr, file))
  {
    return -1;
  }

  hdr->datarecords++;

  fflush(file);

  return 0;
}


static int edflib_write_edf_header(struct edfhdrblock *hdr)
{
  int i, j, p, q,
      len,
      rest,
      edfsignals;

  char str[128];

  struct tm *date_time;

  time_t elapsed_time;

  FILE *file;


  file = hdr->file_hdl;

  edfsignals = hdr->edfsignals;

  if(edfsignals<0)
  {
    return EDFLIB_NO_SIGNALS;
  }

  if(edfsignals>EDFLIB_MAXSIGNALS)
  {
    return EDFLIB_TOO_MANY_SIGNALS;
  }

  hdr->eq_sf = 1;

  hdr->recordsize = 0;

  hdr->total_annot_bytes = EDFLIB_ANNOTATION_BYTES * hdr->nr_annot_chns;

  for(i=0; i<edfsignals; i++)
  {
   if (hdr->edfplus || hdr->bdfplus)
   {
      if(hdr->edfparam[i].smp_per_record<1)
      {
        return EDFLIB_NO_SAMPLES_IN_RECORD ;
      }
  
      if(hdr->edfparam[i].dig_max==hdr->edfparam[i].dig_min)
      {
        return EDFLIB_DIGMIN_IS_DIGMAX;
      }
  
      if(hdr->edfparam[i].dig_max<hdr->edfparam[i].dig_min)
      {
        return EDFLIB_DIGMAX_LOWER_THAN_DIGMIN;
      }
  
      if(hdr->edfparam[i].phys_max==hdr->edfparam[i].phys_min)
      {
        return EDFLIB_PHYSMIN_IS_PHYSMAX;
      }
	  
	  hdr->recordsize += hdr->edfparam[i].smp_per_record;
	  
    }
    if(i > 0)
    {
      if(hdr->edfparam[i].smp_per_record != hdr->edfparam[i-1].smp_per_record)
      {
        hdr->eq_sf = 0;
      }
    }
  }
	  

  if(hdr->edf)
  {
    hdr->recordsize *= 2;

    hdr->recordsize += hdr->total_annot_bytes;

    if(hdr->recordsize > (10 * 1024 * 1024))  /* datarecord size should not exceed 10MB for EDF */
    {
      return EDFLIB_DATARECORD_SIZE_TOO_BIG;
    }  /* if your application gets hit by this limitation, lower the value for the datarecord duration */
       /* using the function edf_set_datarecord_duration() */
  }
  else
  {
    hdr->recordsize *= 3;

    hdr->recordsize += hdr->total_annot_bytes;

    if(hdr->recordsize > (15 * 1024 * 1024))  /* datarecord size should not exceed 15MB for BDF */
    {
      return EDFLIB_DATARECORD_SIZE_TOO_BIG;
    }  /* if your application gets hit by this limitation, lower the value for the datarecord duration */
       /* using the function edf_set_datarecord_duration() */
  }

  for(i=0; i<edfsignals; i++)
  {
	if ((hdr->edfparam[i].phys_max == hdr->edfparam[i].phys_min) || (hdr->edfparam[i].dig_max == hdr->edfparam[i].dig_min))
	{
		hdr->edfparam[i].bitvalue = 1;
		hdr->edfparam[i].offset = 0;
	}
	else
	{
    hdr->edfparam[i].bitvalue = (hdr->edfparam[i].phys_max - hdr->edfparam[i].phys_min) / (hdr->edfparam[i].dig_max - hdr->edfparam[i].dig_min);
    hdr->edfparam[i].offset = hdr->edfparam[i].phys_max / hdr->edfparam[i].bitvalue - hdr->edfparam[i].dig_max;
	}
  }

  rewind(file);

  if(hdr->edf)
  {
    fprintf(file, "0       ");
  }
  else
  {
    fputc(255, file);
    fprintf(file, "BIOSEMI");
  }

  p = 0;

  if(hdr->plus_birthdate[0]==0)
  {
    rest = 72;
  }
  else
  {
    rest = 62;
  }

  len = strlen(hdr->plus_patientcode);
  if(len && rest)
  {
    if(len>rest)
    {
      len = rest;
      rest = 0;
    }
    else
    {
      rest -= len;
    }
    edflib_strlcpy(str, hdr->plus_patientcode, 128);
    edflib_latin1_to_ascii(str, len);
    str[len] = 0;
    for(i=0; i<len; i++)
    {
      if(str[i]==' ')
      {
        str[i] = '_';
      }
    }
    p += fprintf(file, "%s ", str);
  }
  else
  {
    p += fprintf(file, "X ");
  }

  if(hdr->plus_gender[0]=='M')
  {
    fputc('M', file);
  }
  else
  {
    if(hdr->plus_gender[0]=='F')
    {
      fputc('F', file);
    }
    else
    {
      fputc('X', file);
    }
  }
  fputc(' ', file);
  p +=2;

  if(hdr->plus_birthdate[0]==0)
  {
    fputc('X', file);
    fputc(' ', file);

    p +=2;
  }
  else
  {
    fputc(hdr->plus_birthdate[0], file);
    fputc(hdr->plus_birthdate[1], file);
    fputc('-', file);
    q = edflib_atof_nonlocalized(&(hdr->plus_birthdate[3]));
    switch(q)
    {
      case  1: fprintf(file, "JAN");  break;
      case  2: fprintf(file, "FEB");  break;
      case  3: fprintf(file, "MAR");  break;
      case  4: fprintf(file, "APR");  break;
      case  5: fprintf(file, "MAY");  break;
      case  6: fprintf(file, "JUN");  break;
      case  7: fprintf(file, "JUL");  break;
      case  8: fprintf(file, "AUG");  break;
      case  9: fprintf(file, "SEP");  break;
      case 10: fprintf(file, "OCT");  break;
      case 11: fprintf(file, "NOV");  break;
      case 12: fprintf(file, "DEC");  break;
    }
    fputc('-', file);
    fputc(hdr->plus_birthdate[6], file);
    fputc(hdr->plus_birthdate[7], file);
    fputc(hdr->plus_birthdate[8], file);
    fputc(hdr->plus_birthdate[9], file);
    fputc(' ', file);

    p += 12;
  }

  len = strlen(hdr->plus_patient_name);
  if(len && rest)
  {
    if(len>rest)
    {
      len = rest;
      rest = 0;
    }
    else
    {
      rest -= len;
    }
    edflib_strlcpy(str, hdr->plus_patient_name, 128);
    edflib_latin1_to_ascii(str, len);
    str[len] = 0;
    for(i=0; i<len; i++)
    {
      if(str[i]==' ')
      {
        str[i] = '_';
      }
    }
    p += fprintf(file, "%s", str);
  }
  else
  {
    fputc('X', file);

    p++;
  }

  if(rest)
  {
    fputc(' ', file);

    p++;

    rest--;
  }

  len = strlen(hdr->plus_patient_additional);
  if(len && rest)
  {
    if(len>rest)
    {
      len = rest;
    }
    edflib_strlcpy(str, hdr->plus_patient_additional, 128);
    edflib_latin1_to_ascii(str, len);
    str[len] = 0;
    p += fprintf(file, "%s", str);
  }

  for(; p<80; p++)
  {
    fputc(' ', file);
  }

  if(!hdr->startdate_year)
  {
    elapsed_time = time(NULL);
    date_time = localtime(&elapsed_time);

    hdr->startdate_year = date_time->tm_year + 1900;
    hdr->startdate_month = date_time->tm_mon + 1;
    hdr->startdate_day = date_time->tm_mday;
    hdr->starttime_hour = date_time->tm_hour;
    hdr->starttime_minute = date_time->tm_min;
    hdr->starttime_second = date_time->tm_sec % 60;
  }

  p = 0;

  p += fprintf(file, "Startdate %02u-", hdr->startdate_day);
  switch(hdr->startdate_month)
  {
    case  1 : fprintf(file, "JAN");  break;
    case  2 : fprintf(file, "FEB");  break;
    case  3 : fprintf(file, "MAR");  break;
    case  4 : fprintf(file, "APR");  break;
    case  5 : fprintf(file, "MAY");  break;
    case  6 : fprintf(file, "JUN");  break;
    case  7 : fprintf(file, "JUL");  break;
    case  8 : fprintf(file, "AUG");  break;
    case  9 : fprintf(file, "SEP");  break;
    case 10 : fprintf(file, "OCT");  break;
    case 11 : fprintf(file, "NOV");  break;
    case 12 : fprintf(file, "DEC");  break;
  }
  p += 3;
  fputc('-', file);
  p++;
  p += edflib_fprint_int_number_nonlocalized(file, hdr->startdate_year, 4, 0);
  fputc(' ', file);
  p++;

  rest = 42;

  len = strlen(hdr->plus_admincode);
  if(len && rest)
  {
    if(len>rest)
    {
      len = rest;
      rest = 0;
    }
    else
    {
      rest -= len;
    }
    edflib_strlcpy(str, hdr->plus_admincode, 128);
    edflib_latin1_to_ascii(str, len);
    str[len] = 0;
    for(i=0; i<len; i++)
    {
      if(str[i]==' ')
      {
        str[i] = '_';
      }
    }
    p += fprintf(file, "%s", str);
  }
  else
  {
    p += fprintf(file, "X");
  }

  if(rest)
  {
    fputc(' ', file);

    p++;

    rest--;
  }

  len = strlen(hdr->plus_technician);
  if(len && rest)
  {
    if(len>rest)
    {
      len = rest;
      rest = 0;
    }
    else
    {
      rest -= len;
    }
    edflib_strlcpy(str, hdr->plus_technician, 128);
    edflib_latin1_to_ascii(str, len);
    str[len] = 0;
    for(i=0; i<len; i++)
    {
      if(str[i]==' ')
      {
        str[i] = '_';
      }
    }
    p += fprintf(file, "%s", str);
  }
  else
  {
    p += fprintf(file, "X");
  }

  if(rest)
  {
    fputc(' ', file);

    p++;

    rest--;
  }

  len = strlen(hdr->plus_equipment);
  if(len && rest)
  {
    if(len>rest)
    {
      len = rest;
      rest = 0;
    }
    else
    {
      rest -= len;
    }
    edflib_strlcpy(str, hdr->plus_equipment, 128);
    edflib_latin1_to_ascii(str, len);
    str[len] = 0;
    for(i=0; i<len; i++)
    {
      if(str[i]==' ')
      {
        str[i] = '_';
      }
    }
    p += fprintf(file, "%s", str);
  }
  else
  {
    p += fprintf(file, "X");
  }

  if(rest)
  {
    fputc(' ', file);

    p++;

    rest--;
  }

  len = strlen(hdr->plus_recording_additional);
  if(len && rest)
  {
    if(len>rest)
    {
      len = rest;
    }
    edflib_strlcpy(str, hdr->plus_recording_additional, 128);
    edflib_latin1_to_ascii(str, len);
    str[len] = 0;
    p += fprintf(file, "%s", str);
  }

  for(; p<80; p++)
  {
    fputc(' ', file);
  }

  fprintf(file, "%02u.%02u.%02u", hdr->startdate_day, hdr->startdate_month, (hdr->startdate_year % 100));
  fprintf(file, "%02u.%02u.%02u", hdr->starttime_hour, hdr->starttime_minute, hdr->starttime_second);
  p = edflib_fprint_int_number_nonlocalized(file, (edfsignals + hdr->nr_annot_chns + 1) * 256, 0, 0);
  for(; p<8; p++)
  {
    fputc(' ', file);
  }
  if(hdr->edfplus)
  {
    fprintf(file, "EDF+C");
  }
  else if(hdr->bdfplus)
  {
    fprintf(file, "BDF+C");
  }
  else
  {
    fprintf(file, "     ");
  }
  for(i=0; i<39; i++)
  {
    fputc(' ', file);
  }
  fprintf(file, "-1      ");
  if(hdr->long_data_record_duration == EDFLIB_TIME_DIMENSION)
  {
    fprintf(file, "1       ");
  }
  else
  {
    edflib_snprint_number_nonlocalized(str, hdr->data_record_duration, 128);
    edflib_strlcat(str, "        ", 128);
    str[8] = 0;
    fprintf(file, "%s", str);
  }
  p = edflib_fprint_int_number_nonlocalized(file, edfsignals + hdr->nr_annot_chns, 0, 0);
  for(; p<4; p++)
  {
    fputc(' ', file);
  }

  for(i=0; i<edfsignals; i++)
  {
    len = strlen(hdr->edfparam[i].label);
    edflib_latin1_to_ascii(hdr->edfparam[i].label, len);
    for(j=0; j<len; j++)
    {
      fputc(hdr->edfparam[i].label[j], file);
    }
    for(; j<16; j++)
    {
      fputc(' ', file);
    }
  }
  for(j=0; j<hdr->nr_annot_chns; j++)
  {
    if(hdr->edf)
    {
      fprintf(file, "EDF Annotations ");
    }
    else
    {
      fprintf(file, "BDF Annotations ");
    }
  }
  for(i=0; i<edfsignals; i++)
  {
    len = strlen(hdr->edfparam[i].transducer);
    edflib_latin1_to_ascii(hdr->edfparam[i].transducer, len);
    for(j=0; j<len; j++)
    {
      fputc(hdr->edfparam[i].transducer[j], file);
    }
    for(; j<80; j++)
    {
      fputc(' ', file);
    }
  }
  for(j=0; j<hdr->nr_annot_chns; j++)
  {
    for(i=0; i<80; i++)
    {
      fputc(' ', file);
    }
  }
  for(i=0; i<edfsignals; i++)
  {
    len = strlen(hdr->edfparam[i].physdimension);
    edflib_latin1_to_ascii(hdr->edfparam[i].physdimension, len);
    for(j=0; j<len; j++)
    {
      fputc(hdr->edfparam[i].physdimension[j], file);
    }
    for(; j<8; j++)
    {
      fputc(' ', file);
    }
  }
  for(j=0; j<hdr->nr_annot_chns; j++)
  {
    fprintf(file, "        ");
  }
  for(i=0; i<edfsignals; i++)
  {
    p = edflib_snprint_number_nonlocalized(str, hdr->edfparam[i].phys_min, 128);
    for(; p<8; p++)
    {
      str[p] = ' ';
    }
    str[8] = 0;
    fprintf(file, "%s", str);
  }
  for(j=0; j<hdr->nr_annot_chns; j++)
  {
    fprintf(file, "-1      ");
  }
  for(i=0; i<edfsignals; i++)
  {
    p = edflib_snprint_number_nonlocalized(str, hdr->edfparam[i].phys_max, 128);
    for(; p<8; p++)
    {
      str[p] = ' ';
    }
    str[8] = 0;
    fprintf(file, "%s", str);
  }
  for(j=0; j<hdr->nr_annot_chns; j++)
  {
    fprintf(file, "1       ");
  }
  for(i=0; i<edfsignals; i++)
  {
    p = edflib_fprint_int_number_nonlocalized(file, hdr->edfparam[i].dig_min, 0, 0);
    for(; p<8; p++)
    {
      fputc(' ', file);
    }
  }
  for(j=0; j<hdr->nr_annot_chns; j++)
  {
    if(hdr->edf)
    {
      fprintf(file, "-32768  ");
    }
    else
    {
      fprintf(file, "-8388608");
    }
  }
  for(i=0; i<edfsignals; i++)
  {
    p = edflib_fprint_int_number_nonlocalized(file, hdr->edfparam[i].dig_max, 0, 0);
    for(; p<8; p++)
    {
      fputc(' ', file);
    }
  }
  for(j=0; j<hdr->nr_annot_chns; j++)
  {
    if(hdr->edf)
    {
      fprintf(file, "32767   ");
    }
    else
    {
      fprintf(file, "8388607 ");
    }
  }
  for(i=0; i<edfsignals; i++)
  {
    len = strlen(hdr->edfparam[i].prefilter);
    edflib_latin1_to_ascii(hdr->edfparam[i].prefilter, len);
    for(j=0; j<len; j++)
    {
      fputc(hdr->edfparam[i].prefilter[j], file);
    }
    for(; j<80; j++)
    {
      fputc(' ', file);
    }
  }
  for(i=0; i<hdr->nr_annot_chns; i++)
  {
    for(j=0; j<80; j++)
    {
      fputc(' ', file);
    }
  }
  for(i=0; i<edfsignals; i++)
  {
    p = edflib_fprint_int_number_nonlocalized(file, hdr->edfparam[i].smp_per_record, 0, 0);
    for(; p<8; p++)
    {
      fputc(' ', file);
    }
  }
  for(j=0; j<hdr->nr_annot_chns; j++)
  {
    if(hdr->edf)
    {
      p = edflib_fprint_int_number_nonlocalized(file, EDFLIB_ANNOTATION_BYTES / 2, 0, 0);
      for(; p<8; p++)
      {
        fputc(' ', file);
      }
    }
    else
    {
      p = edflib_fprint_int_number_nonlocalized(file, EDFLIB_ANNOTATION_BYTES / 3, 0, 0);
      for(; p<8; p++)
      {
        fputc(' ', file);
      }
    }
  }
  for(i=0; i<(edfsignals * 32); i++)
  {
    fputc(' ', file);
  }
  for(i=0; i<(hdr->nr_annot_chns * 32); i++)
  {
    fputc(' ', file);
  }
  
  return 0;
}


int edf_set_label(int handle, int edfsignal, const char *label)
{
  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  if(!(hdrlist[handle]->writemode))
  {
    return -1;
  }

  if(edfsignal<0)
  {
    return -1;
  }

  if(edfsignal>=hdrlist[handle]->edfsignals)
  {
    return -1;
  }

  if(hdrlist[handle]->datarecords)
  {
    return -1;
  }

  strncpy(hdrlist[handle]->edfparam[edfsignal].label, label, 16);

  hdrlist[handle]->edfparam[edfsignal].label[16] = 0;

  edflib_remove_padding_trailing_spaces(hdrlist[handle]->edfparam[edfsignal].label);

  return 0;
}


int edf_set_physical_dimension(int handle, int edfsignal, const char *phys_dim)
{
  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  if(!(hdrlist[handle]->writemode))
  {
    return -1;
  }

  if(edfsignal<0)
  {
    return -1;
  }

  if(edfsignal>=hdrlist[handle]->edfsignals)
  {
    return -1;
  }

  if(hdrlist[handle]->datarecords)
  {
    return -1;
  }

  strncpy(hdrlist[handle]->edfparam[edfsignal].physdimension, phys_dim, 8);

  hdrlist[handle]->edfparam[edfsignal].physdimension[8] = 0;

  edflib_remove_padding_trailing_spaces(hdrlist[handle]->edfparam[edfsignal].physdimension);

  return 0;
}


int edf_set_physical_maximum(int handle, int edfsignal, double phys_max)
{
  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  if(!(hdrlist[handle]->writemode))
  {
    return -1;
  }

  if(edfsignal<0)
  {
    return -1;
  }

  if(edfsignal>=hdrlist[handle]->edfsignals)
  {
    return -1;
  }

  if(hdrlist[handle]->datarecords)
  {
    return -1;
  }

  hdrlist[handle]->edfparam[edfsignal].phys_max = phys_max;

  return 0;
}


int edf_set_physical_minimum(int handle, int edfsignal, double phys_min)
{
  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  if(!(hdrlist[handle]->writemode))
  {
    return -1;
  }

  if(edfsignal<0)
  {
    return -1;
  }

  if(edfsignal>=hdrlist[handle]->edfsignals)
  {
    return -1;
  }

  if(hdrlist[handle]->datarecords)
  {
    return -1;
  }

  hdrlist[handle]->edfparam[edfsignal].phys_min = phys_min;

  return 0;
}


int edf_set_digital_maximum(int handle, int edfsignal, int dig_max)
{
  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  if(!(hdrlist[handle]->writemode))
  {
    return -1;
  }

  if(edfsignal<0)
  {
    return -1;
  }

  if(edfsignal>=hdrlist[handle]->edfsignals)
  {
    return -1;
  }

  if(hdrlist[handle]->edf)
  {
    if(dig_max > 32767)
    {
      return -1;
    }
  }
  else
  {
    if(dig_max > 8388607)
    {
      return -1;
    }
  }

  if(hdrlist[handle]->datarecords)
  {
    return -1;
  }

  hdrlist[handle]->edfparam[edfsignal].dig_max = dig_max;

  return 0;
}


int edf_set_digital_minimum(int handle, int edfsignal, int dig_min)
{
  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  if(!(hdrlist[handle]->writemode))
  {
    return -1;
  }

  if(edfsignal<0)
  {
    return -1;
  }

  if(edfsignal>=hdrlist[handle]->edfsignals)
  {
    return -1;
  }

  if(hdrlist[handle]->edf)
  {
    if(dig_min < (-32768))
    {
      return -1;
    }
  }
  else
  {
    if(dig_min < (-8388608))
    {
      return -1;
    }
  }

  if(hdrlist[handle]->datarecords)
  {
    return -1;
  }

  hdrlist[handle]->edfparam[edfsignal].dig_min = dig_min;

  return 0;
}


int edf_set_patientname(int handle, const char *patientname)
{
  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  if(!(hdrlist[handle]->writemode))
  {
    return -1;
  }

  if(hdrlist[handle]->datarecords)
  {
    return -1;
  }

  strncpy(hdrlist[handle]->plus_patient_name, patientname, 80);

  hdrlist[handle]->plus_patient_name[80] = 0;

  edflib_remove_padding_trailing_spaces(hdrlist[handle]->plus_patient_name);

  return 0;
}


int edf_set_patientcode(int handle, const char *patientcode)
{
  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  if(!(hdrlist[handle]->writemode))
  {
    return -1;
  }

  if(hdrlist[handle]->datarecords)
  {
    return -1;
  }

  strncpy(hdrlist[handle]->plus_patientcode, patientcode, 80);

  hdrlist[handle]->plus_patientcode[80] = 0;

  edflib_remove_padding_trailing_spaces(hdrlist[handle]->plus_patientcode);

  return 0;
}


int edf_set_gender(int handle, int gender)
{
  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  if(!(hdrlist[handle]->writemode))
  {
    return -1;
  }

  if(hdrlist[handle]->datarecords)
  {
    return -1;
  }

  if((gender<0)||(gender>1))
  {
    return -1;
  }

  if(gender)
  {
    hdrlist[handle]->plus_gender[0] = 'M';
  }
  else
  {
    hdrlist[handle]->plus_gender[0] = 'F';
  }

  hdrlist[handle]->plus_gender[1] = 0;

  return 0;
}


int edf_set_birthdate(int handle, int birthdate_year, int birthdate_month, int birthdate_day)
{
  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  if(!(hdrlist[handle]->writemode))
  {
    return -1;
  }

  if(hdrlist[handle]->datarecords)
  {
    return -1;
  }

  if((birthdate_year<1800) || (birthdate_year>3000) ||
     (birthdate_month<1)   || (birthdate_month>12)  ||
     (birthdate_day<1)     || (birthdate_day>31))
  {
    return -1;
  }

  sprintf(hdrlist[handle]->plus_birthdate, "%02i.%02i.%02i%02i", birthdate_day, birthdate_month, birthdate_year / 100, birthdate_year % 100);

  hdrlist[handle]->plus_birthdate[10] = 0;

  return 0;
}


int edf_set_patient_additional(int handle, const char *patient_additional)
{
  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  if(!(hdrlist[handle]->writemode))
  {
    return -1;
  }

  if(hdrlist[handle]->datarecords)
  {
    return -1;
  }

  strncpy(hdrlist[handle]->plus_patient_additional, patient_additional, 80);

  hdrlist[handle]->plus_patient_additional[80] = 0;

  edflib_remove_padding_trailing_spaces(hdrlist[handle]->plus_patient_additional);

  return 0;
}


int edf_set_admincode(int handle, const char *admincode)
{
  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  if(!(hdrlist[handle]->writemode))
  {
    return -1;
  }

  if(hdrlist[handle]->datarecords)
  {
    return -1;
  }

  strncpy(hdrlist[handle]->plus_admincode, admincode, 80);

  hdrlist[handle]->plus_admincode[80] = 0;

  edflib_remove_padding_trailing_spaces(hdrlist[handle]->plus_admincode);

  return 0;
}


int edf_set_technician(int handle, const char *technician)
{
  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  if(!(hdrlist[handle]->writemode))
  {
    return -1;
  }

  if(hdrlist[handle]->datarecords)
  {
    return -1;
  }

  strncpy(hdrlist[handle]->plus_technician, technician, 80);

  hdrlist[handle]->plus_technician[80] = 0;

  edflib_remove_padding_trailing_spaces(hdrlist[handle]->plus_technician);

  return 0;
}


int edf_set_equipment(int handle, const char *equipment)
{
  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  if(!(hdrlist[handle]->writemode))
  {
    return -1;
  }

  if(hdrlist[handle]->datarecords)
  {
    return -1;
  }

  strncpy(hdrlist[handle]->plus_equipment, equipment, 80);

  hdrlist[handle]->plus_equipment[80] = 0;

  edflib_remove_padding_trailing_spaces(hdrlist[handle]->plus_equipment);

  return 0;
}


int edf_set_recording_additional(int handle, const char *recording_additional)
{
  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  if(!(hdrlist[handle]->writemode))
  {
    return -1;
  }

  if(hdrlist[handle]->datarecords)
  {
    return -1;
  }

  strncpy(hdrlist[handle]->plus_recording_additional, recording_additional, 80);

  hdrlist[handle]->plus_recording_additional[80] = 0;

  edflib_remove_padding_trailing_spaces(hdrlist[handle]->plus_recording_additional);

  return 0;
}


int edf_set_startdatetime(int handle, int startdate_year, int startdate_month, int startdate_day,
                                      int starttime_hour, int starttime_minute, int starttime_second)
{
  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  if(!(hdrlist[handle]->writemode))
  {
    return -1;
  }

  if(hdrlist[handle]->datarecords)
  {
    return -1;
  }

  if((startdate_year<1970) || (startdate_year>3000) ||
     (startdate_month<1)   || (startdate_month>12)  ||
     (startdate_day<1)     || (startdate_day>31)    ||
     (starttime_hour<0)    || (starttime_hour>23)   ||
     (starttime_minute<0)  || (starttime_minute>59) ||
     (starttime_second<0)  || (starttime_second>59))
  {
    return -1;
  }

  hdrlist[handle]->startdate_year = startdate_year;
  hdrlist[handle]->startdate_month = startdate_month;
  hdrlist[handle]->startdate_day = startdate_day;
  hdrlist[handle]->starttime_hour = starttime_hour;
  hdrlist[handle]->starttime_minute = starttime_minute;
  hdrlist[handle]->starttime_second = starttime_second;

  return 0;
}


int edfwrite_annotation_utf8(int handle, long long onset, long long duration, const char *description)
{
  int i;

  struct edf_write_annotationblock *list_annot, *malloc_list;


  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  if(!(hdrlist[handle]->writemode))
  {
    return -1;
  }

  if(onset<0LL)
  {
    return -1;
  }

  if(hdrlist[handle]->annots_in_file >= hdrlist[handle]->annotlist_sz)
  {
    malloc_list = (struct edf_write_annotationblock *)realloc(write_annotationslist[handle],
                                                              sizeof(struct edf_write_annotationblock) * (hdrlist[handle]->annotlist_sz + EDFLIB_ANNOT_MEMBLOCKSZ));
    if(malloc_list==NULL)
    {
      return -1;
    }

    write_annotationslist[handle] = malloc_list;

    hdrlist[handle]->annotlist_sz += EDFLIB_ANNOT_MEMBLOCKSZ;
  }

  list_annot = write_annotationslist[handle] + hdrlist[handle]->annots_in_file;

  list_annot->onset = onset;
  list_annot->duration = duration;
  strncpy(list_annot->annotation, description, EDFLIB_WRITE_MAX_ANNOTATION_LEN);
  list_annot->annotation[EDFLIB_WRITE_MAX_ANNOTATION_LEN] = 0;

  for(i=0; ; i++)
  {
    if(list_annot->annotation[i] == 0)
    {
      break;
    }

    if(list_annot->annotation[i] < 32)
    {
      list_annot->annotation[i] = '.';
    }
  }

  hdrlist[handle]->annots_in_file++;

  return 0;
}


int edfwrite_annotation_latin1(int handle, long long onset, long long duration, const char *description)
{
  struct edf_write_annotationblock *list_annot, *malloc_list;

  char str[EDFLIB_WRITE_MAX_ANNOTATION_LEN + 1];


  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  if(!(hdrlist[handle]->writemode))
  {
    return -1;
  }

  if(onset<0LL)
  {
    return -1;
  }

  if(hdrlist[handle]->annots_in_file >= hdrlist[handle]->annotlist_sz)
  {
    malloc_list = (struct edf_write_annotationblock *)realloc(write_annotationslist[handle],
                                                              sizeof(struct edf_write_annotationblock) * (hdrlist[handle]->annotlist_sz + EDFLIB_ANNOT_MEMBLOCKSZ));
    if(malloc_list==NULL)
    {
      return -1;
    }

    write_annotationslist[handle] = malloc_list;

    hdrlist[handle]->annotlist_sz += EDFLIB_ANNOT_MEMBLOCKSZ;
  }

  list_annot = write_annotationslist[handle] + hdrlist[handle]->annots_in_file;

  list_annot->onset = onset;
  list_annot->duration = duration;
  strncpy(str, description, EDFLIB_WRITE_MAX_ANNOTATION_LEN);
  str[EDFLIB_WRITE_MAX_ANNOTATION_LEN] = 0;
  edflib_latin12utf8(str, strlen(str));
  strncpy(list_annot->annotation, str, EDFLIB_WRITE_MAX_ANNOTATION_LEN);
  list_annot->annotation[EDFLIB_WRITE_MAX_ANNOTATION_LEN] = 0;

  hdrlist[handle]->annots_in_file++;

  return 0;
}


static void edflib_remove_padding_trailing_spaces(char *str)
{
  int i;

  while(str[0]==' ')
  {
    for(i=0; ; i++)
    {
      if(str[i]==0)
      {
        break;
      }

      str[i] = str[i+1];
    }
  }

  for(i = strlen(str); i>0; i--)
  {
    if(str[i-1]==' ')
    {
      str[i-1] = 0;
    }
    else
    {
      break;
    }
  }
}


int edf_set_prefilter(int handle, int edfsignal, const char *prefilter)
{
  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  if(!(hdrlist[handle]->writemode))
  {
    return -1;
  }

  if(edfsignal<0)
  {
    return -1;
  }

  if(edfsignal>=hdrlist[handle]->edfsignals)
  {
    return -1;
  }

  if(hdrlist[handle]->datarecords)
  {
    return -1;
  }

  strncpy(hdrlist[handle]->edfparam[edfsignal].prefilter, prefilter, 80);

  hdrlist[handle]->edfparam[edfsignal].prefilter[80] = 0;

  edflib_remove_padding_trailing_spaces(hdrlist[handle]->edfparam[edfsignal].prefilter);

  return 0;
}


int edf_set_transducer(int handle, int edfsignal, const char *transducer)
{
  if(handle<0)
  {
    return -1;
  }

  if(handle>=EDFLIB_MAXFILES)
  {
    return -1;
  }

  if(hdrlist[handle]==NULL)
  {
    return -1;
  }

  if(!(hdrlist[handle]->writemode))
  {
    return -1;
  }

  if(edfsignal<0)
  {
    return -1;
  }

  if(edfsignal>=hdrlist[handle]->edfsignals)
  {
    return -1;
  }

  if(hdrlist[handle]->datarecords)
  {
    return -1;
  }

  strncpy(hdrlist[handle]->edfparam[edfsignal].transducer, transducer, 80);

  hdrlist[handle]->edfparam[edfsignal].transducer[80] = 0;

  edflib_remove_padding_trailing_spaces(hdrlist[handle]->edfparam[edfsignal].transducer);

  return 0;
}


/* minimum is the minimum digits that will be printed (minus sign not included), leading zero's will be added if necessary */
/* if sign is zero, only negative numbers will have the sign '-' character */
/* if sign is one, the sign '+' or '-' character will always be printed */
/* returns the amount of characters printed */
static int edflib_fprint_int_number_nonlocalized(FILE *file, int q, int minimum, int sign)
{
  int flag=0, z, i, j=0, base = 1000000000;

  if(minimum < 0)
  {
    minimum = 0;
  }

  if(minimum > 9)
  {
    flag = 1;
  }

  if(q < 0)
  {
    fputc('-', file);

    j++;

    q = -q;
  }
  else
  {
    if(sign)
    {
      fputc('+', file);

      j++;
    }
  }

  for(i=10; i; i--)
  {
    if(minimum == i)
    {
      flag = 1;
    }

    z = q / base;

    q %= base;

    if(z || flag)
    {
      fputc('0' + z, file);

      j++;

      flag = 1;
    }

    base /= 10;
  }

  if(!flag)
  {
    fputc('0', file);

    j++;
  }

  return j;
}


/* minimum is the minimum digits that will be printed (minus sign not included), leading zero's will be added if necessary */
/* if sign is zero, only negative numbers will have the sign '-' character */
/* if sign is one, the sign '+' or '-' character will always be printed */
/* returns the amount of characters printed */
static int edflib_fprint_ll_number_nonlocalized(FILE *file, long long q, int minimum, int sign)
{
  int flag=0, z, i, j=0;

  long long base = 1000000000000000000LL;

  if(minimum < 0)
  {
    minimum = 0;
  }

  if(minimum > 18)
  {
    flag = 1;
  }

  if(q < 0LL)
  {
    fputc('-', file);

    j++;

    q = -q;
  }
  else
  {
    if(sign)
    {
      fputc('+', file);

      j++;
    }
  }

  for(i=19; i; i--)
  {
    if(minimum == i)
    {
      flag = 1;
    }

    z = q / base;

    q %= base;

    if(z || flag)
    {
      fputc('0' + z, file);

      j++;

      flag = 1;
    }

    base /= 10LL;
  }

  if(!flag)
  {
    fputc('0', file);

    j++;
  }

  return j;
}


/* minimum is the minimum digits that will be printed (minus sign not included), leading zero's will be added if necessary */
/* if sign is zero, only negative numbers will have the sign '-' character */
/* if sign is one, the sign '+' or '-' character will always be printed */
/* returns the amount of characters printed */
/*
static int edflib_sprint_int_number_nonlocalized(char *str, int q, int minimum, int sign)
{
  int flag=0, z, i, j=0, base = 1000000000;

  if(minimum < 0)
  {
    minimum = 0;
  }

  if(minimum > 9)
  {
    flag = 1;
  }

  if(q < 0)
  {
    str[j++] = '-';

    q = -q;
  }
  else
  {
    if(sign)
    {
      str[j++] = '+';
    }
  }

  for(i=10; i; i--)
  {
    if(minimum == i)
    {
      flag = 1;
    }

    z = q / base;

    q %= base;

    if(z || flag)
    {
      str[j++] = '0' + z;

      flag = 1;
    }

    base /= 10;
  }

  if(!flag)
  {
    str[j++] = '0';
  }

  str[j] = 0;

  return j;
}
*/

/* minimum is the minimum digits that will be printed (minus sign not included), leading zero's will be added if necessary */
/* if sign is zero, only negative numbers will have the sign '-' character */
/* if sign is one, the sign '+' or '-' character will always be printed */
/* returns the amount of characters printed */
static int edflib_snprint_ll_number_nonlocalized(char *dest, long long q, int minimum, int sign, int sz)
{
  int flag=0, z, i, j=0;

  long long base = 1000000000000000000LL;

  if(sz < 1)
  {
    return 0;
  }

  if(minimum < 0)
  {
    minimum = 0;
  }

  if(minimum > 18)
  {
    flag = 1;
  }

  if(q < 0LL)
  {
    dest[j++] = '-';

    q = -q;
  }
  else
  {
    if(sign)
    {
      dest[j++] = '+';
    }
  }

  if(j == sz)
  {
    dest[--j] = 0;

    return j;
  }

  for(i=19; i; i--)
  {
    if(minimum == i)
    {
      flag = 1;
    }

    z = q / base;

    q %= base;

    if(z || flag)
    {
      dest[j++] = '0' + z;

      if(j == sz)
      {
        dest[--j] = 0;

        return j;
      }

      flag = 1;
    }

    base /= 10LL;
  }

  if(!flag)
  {
    dest[j++] = '0';
  }

  if(j == sz)
  {
    dest[--j] = 0;

    return j;
  }

  dest[j] = 0;

  return j;}


static int edflib_snprint_number_nonlocalized(char *dest, double val, int sz)
{
  int flag=0, z, i, j=0, q, base = 1000000000;

  double var;

  if(sz < 1)  return 0;

  q = (int)val;

  var = val - q;

  if(val < 0.0)
  {
    dest[j++] = '-';

    if(q < 0)
    {
      q = -q;
    }
  }

  if(j == sz)
  {
    dest[--j] = 0;

    return j;
  }

  for(i=10; i; i--)
  {
    z = q / base;

    q %= base;

    if(z || flag)
    {
      dest[j++] = '0' + z;

      if(j == sz)
      {
        dest[--j] = 0;

        return j;
      }

      flag = 1;
    }

    base /= 10;
  }

  if(!flag)
  {
    dest[j++] = '0';
  }

  if(j == sz)
  {
    dest[--j] = 0;

    return j;
  }

  base = 100000000;

  var *= (base * 10);

  q = (int)var;

  if(q < 0)
  {
    q = -q;
  }

  if(!q)
  {
    dest[j] = 0;

    return j;
  }

  dest[j++] = '.';

  if(j == sz)
  {
    dest[--j] = 0;

    return j;
  }

  for(i=9; i; i--)
  {
    z = q / base;

    q %= base;

    dest[j++] = '0' + z;

    if(j == sz)
    {
      dest[--j] = 0;

      return j;
    }

    base /= 10;
  }

  dest[j] = 0;

  j--;

  for(; j>0; j--)
  {
    if(dest[j] == '0')
    {
      dest[j] = 0;
    }
    else
    {
      j++;

      break;
    }
  }

  return j;
}


static double edflib_atof_nonlocalized(const char *str)
{
  int i=0, j, dot_pos=-1, decimals=0, sign=1, exp_pos=-1, exp_sign=1, exp_val=0;

  double value, value2=0.0;


  value = edflib_atoi_nonlocalized(str);

  while(str[i] == ' ')
  {
    i++;
  }

  if((str[i] == '+') || (str[i] == '-'))
  {
    if(str[i] == '-')
    {
      sign = -1;
    }

    i++;
  }

  for(; ; i++)
  {
    if(str[i] == 0)
    {
      break;
    }

    if((str[i] == 'e') || (str[i] == 'E'))
    {
      exp_pos = i;

      break;
    }

    if(((str[i] < '0') || (str[i] > '9')) && (str[i] != '.'))
    {
      break;
    }

    if(dot_pos >= 0)
    {
      if((str[i] >= '0') && (str[i] <= '9'))
      {
        decimals++;
      }
      else
      {
        break;
      }
    }

    if(str[i] == '.')
    {
      if(dot_pos < 0)
      {
        dot_pos = i;
      }
    }
  }

  if(decimals)
  {
    value2 = edflib_atoi_nonlocalized(str + dot_pos + 1) * sign;

    i = 1;

    while(decimals--)
    {
      i *= 10;
    }

    value2 /= i;

    value += value2;
  }

  if(exp_pos > 0)
  {
    i = exp_pos + 1;

    if(str[i])
    {
      if(str[i] == '+')
      {
        i++;
      }
      else if(str[i] == '-')
        {
          exp_sign = -1;

          i++;
        }

      if(str[i])
      {
        exp_val = edflib_atoi_nonlocalized(str + i);

        if(exp_val > 0)
        {
          for(j=0; j<exp_val; j++)
          {
            if(exp_sign > 0)
            {
              value *= 10;
            }
            else
            {
              value /= 10;
            }
          }
        }
      }
    }
  }

  return value;
}


static int edflib_atoi_nonlocalized(const char *str)
{
  int i=0, value=0, sign=1;

  while(str[i] == ' ')
  {
    i++;
  }

  if((str[i] == '+') || (str[i] == '-'))
  {
    if(str[i] == '-')
    {
      sign = -1;
    }

    i++;
  }

  for( ; ; i++)
  {
    if(str[i] == 0)
    {
      break;
    }

    if((str[i] < '0') || (str[i] > '9'))
    {
      break;
    }

    value *= 10;

    value += (str[i] - '0');
  }

  return value * sign;
}


static int edflib_write_tal(struct edfhdrblock *hdr, FILE *file)
{
  if ((hdr->edf||hdr->bdf) && !(hdr->edfplus||hdr->bdfplus)){
	 // EDF/BDF =  means no annotations will be written.
     return 0;
  }
  int p;

  char str[EDFLIB_ANNOTATION_BYTES * (EDFLIB_MAX_ANNOTATION_CHANNELS + 1)];

  p = edflib_snprint_ll_number_nonlocalized(str, (hdr->datarecords * hdr->long_data_record_duration + hdr->starttime_offset) / EDFLIB_TIME_DIMENSION, 0, 1, EDFLIB_ANNOTATION_BYTES * (EDFLIB_MAX_ANNOTATION_CHANNELS + 1));
  if((hdr->long_data_record_duration % EDFLIB_TIME_DIMENSION) || (hdr->starttime_offset))
  {
    str[p++] = '.';
    p += edflib_snprint_ll_number_nonlocalized(str + p, (hdr->datarecords * hdr->long_data_record_duration + hdr->starttime_offset) % EDFLIB_TIME_DIMENSION, 7, 0, (EDFLIB_ANNOTATION_BYTES * (EDFLIB_MAX_ANNOTATION_CHANNELS + 1)) - p);
  }
  str[p++] = 20;
  str[p++] = 20;
  for(; p<hdr->total_annot_bytes; p++)
  {
    str[p] = 0;
  }

  if(fwrite(str, hdr->total_annot_bytes, 1, file) != 1)
  {
    return -1;
  }

  return 0;
}


static int edflib_strlcpy(char *dst, const char *src, int sz)
{
  int srclen;

  sz--;

  srclen = strlen(src);

  if(srclen > sz)  srclen = sz;

  memcpy(dst, src, srclen);

  dst[srclen] = 0;

  return srclen;
}


static int edflib_strlcat(char *dst, const char *src, int sz)
{
  int srclen,
      dstlen;

  dstlen = strlen(dst);

  sz -= dstlen + 1;

  if(!sz)  return dstlen;

  srclen = strlen(src);

  if(srclen > sz)  srclen = sz;

  memcpy(dst + dstlen, src, srclen);

  dst[dstlen + srclen] = 0;

  return (dstlen + srclen);
}




























