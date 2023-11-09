#ifdef _WIN32

#include "fopen_utf8.h"

#include <stdlib.h>
#include <windows.h>


wchar_t* utf8_to_utf16(const char* str)
{
    wchar_t* wstr;

    int require = MultiByteToWideChar(CP_UTF8, 0, str, -1, NULL, 0);
    if (require == 0)
        return NULL;

    wstr = malloc(require * sizeof(wchar_t));
    if (wstr == NULL)
        return NULL;

    if (MultiByteToWideChar(CP_UTF8, 0, str, -1, wstr, require) == 0)
    {
        free(wstr);
        return NULL;
    }

    return wstr;
}


FILE* fopen_utf8(const char* filename, const char* mode)
{
    errno_t error;
    FILE* file;
    wchar_t* wfilename;
    wchar_t* wmode;

    wfilename = utf8_to_utf16(filename);
    if (wfilename == NULL)
        return NULL;

    wmode = utf8_to_utf16(mode);
    if (wmode == NULL)
    {
        free(wfilename);
        return NULL;
    }

    error = _wfopen_s(&file, wfilename, wmode);

    free(wfilename);
    free(wmode);

    return error == 0 ? file : NULL;
}

#endif
