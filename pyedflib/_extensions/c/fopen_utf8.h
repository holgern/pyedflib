#ifndef FOPEN_UTF8_H_
#define FOPEN_UTF8_H_

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
FILE* fopen_utf8(const char* filename, const char* mode);
#else
#define fopen_utf8 fopen
#endif

#ifdef __cplusplus
}
#endif

#endif
