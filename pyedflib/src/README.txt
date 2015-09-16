
Copyright (c) 2009, 2010, 2011, 2013, 2014, 2015 Teunis van Beelen
All rights reserved.

email: teuniz@gmail.com

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
     * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.
     * Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY Teunis van Beelen ''AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL Teunis van Beelen BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.




EDFlib is a programming library for C/C++ for reading and writing EDF+ and BDF+ files.
It also reads "old style" EDF and BDF files.
EDF means European Data Format. BDF is the 24-bits version of EDF.

The library consists of only two files: edflib.h and edflib.c.
The file main.c can be used to compile a program that demonstrates the working and
capabilities of the library. Use it as follows:

test_edflib <filename> <signalnumber>

The program will print the properties of the EDF/BDF-header, the annotations and
the values of 200 samples of the chosen signal.

The file sine.c is another programming example. It will create a BDF+ file containing
the signal "sine", a 1 Hz sinoidal waveform with a samplefrequency of 2048 Hz.

The file test_generator.c shows how to use most of the functions provided by the library.

To view the results use EDFbrowser:  http://www.teuniz.net/edfbrowser/


In order to use this library in your project, copy the files edf.h and edf.c to
your project. Include the file edflib.h in every sourcefile from where you want
to access the library. Don't forget to tell your compiler that it must compile
and link edflib.c (add it to your makefile or buildscript).
edflib.c needs to be compiled with the options "-D_LARGEFILE64_SOURCE -D_LARGEFILE_SOURCE".

For example:

gcc main.c edflib.c -Wall -Wextra -Wshadow -Wformat-nonliteral -Wformat-security -Wtype-limits -g -D_LARGEFILE64_SOURCE -D_LARGEFILE_SOURCE -o test_edflib

gcc sine.c edflib.c -Wall -Wextra -Wshadow -Wformat-nonliteral -Wformat-security -Wtype-limits -g -D_LARGEFILE64_SOURCE -D_LARGEFILE_SOURCE -lm -o sine

gcc test_generator.c edflib.c -Wall -Wextra -Wshadow -Wformat-nonliteral -Wformat-security -Wtype-limits -g -D_LARGEFILE64_SOURCE -D_LARGEFILE_SOURCE -lm -o testgenerator

gcc sweep.c edflib.c -Wall -Wextra -Wshadow -Wformat-nonliteral -Wformat-security -Wtype-limits -g -D_LARGEFILE64_SOURCE -D_LARGEFILE_SOURCE -lm -o sweep


To understand how to use the library, read the comments in the file edflib.h.

The library has been tested using the GCC compiler on Linux and Mingw-w64 on windows.



