# EDFlib

EDFlib is a programming library for C/C++ for reading and writing EDF+ and BDF+ files.
It also reads "old style" EDF and BDF files.
EDF means European Data Format. BDF is the 24-bits version of EDF.


## Usage

The library consists of only two files: `edflib.h` and `edflib.c`.

In order to use EDFlib, copy these two files to your project.
Include the file `edflib.h` in every source file from where you want to access the library.

Don't forget to tell your compiler that it must compile and link `edflib.c` (add it to
your makefile or buildscript). `edflib.c` needs to be compiled with the options
`-D_LARGEFILE64_SOURCE -D_LARGEFILE_SOURCE`.

For example:

`gcc -Wall -Wextra -Wshadow -Wformat-nonliteral -Wformat-security -D_LARGEFILE64_SOURCE -D_LARGEFILE_SOURCE test_edflib.c edflib.c -lm -o test_edflib`

Compilation has been tested using GCC on Linux, Mingw-w64 on Windows, and LLVM GCC on OS X (Yosemite).

To understand how to use the library, read the comments in `edflib.h`.


## Examples

To build the examples: `make`

Each "generator" example creates an EDF+ or BDF+ file with sample signals.

`test_generator` shows how to use most of the functions provided by the library and generates an
EDF+ or BDF+ testfile with several sample signals.

`sine_generator` creates a BDF+ file containing the signal "sine", a 1 Hz sinusoidal waveform with a
sample frequency of 2048 Hz.

`sweep_generator` creates a linear or logarithmic sweep through a range of frequencies in EDF+ or
BDF+ format.

Use EDFbrowser to view these files: https://www.teuniz.net/edfbrowser/

`test_edflib <filename> <signalnumber>` will print the properties of the EDF/BDF header, the
annotations, and the values of 200 samples of the chosen signal. For example, running
`test_generator` will produce the file `test_generator.edf`. Running `test_edflib test_generator.edf 6`
will show the header and first 200 samples of the "noise" signal:
`75  6  27  77  37  30  35  96  62  69  34  15  51  56  69  68  80  45 ...`


## License

Copyright (c) 2009, 2010, 2011, 2013, 2014, 2015 Teunis van Beelen
              2017 Holger Nahrstaedt
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
