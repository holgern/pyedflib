#!/usr/bin/env python

import os
import sys
import sysconfig
from functools import partial

import setuptools
from setuptools import Extension, setup
from setuptools.command.develop import develop

try:
    from Cython.Build import cythonize

    USE_CYTHON = True
except ImportError as e:
    USE_CYTHON = False
    if not os.path.exists(os.path.join("pyedflib", "_extensions", "_pyedflib.c")):
        msg = "Cython must be installed when working with a development version of PyEDFlib"
        raise RuntimeError(msg) from e


# Version of Numpy required for setup
REQUIRED_NUMPY = "numpy>=1.9.1"


# from MDAnalysis setup.py (https://www.mdanalysis.org/)
class NumpyExtension(Extension):
    """Derived class to cleanly handle setup-time (numpy) dependencies."""

    # The only setup-time numpy dependency comes when setting up its
    #  include dir.
    # The actual numpy import and call can be delayed until after pip
    #  has figured it must install numpy.
    # This is accomplished by passing the get_numpy_include function
    #  as one of the include_dirs. This derived Extension class takes
    #  care of calling it when needed.
    def __init__(self, *args, **kwargs):
        self._np_include_dirs = []
        super().__init__(*args, **kwargs)

    @property
    def include_dirs(self):
        if not self._np_include_dirs:
            for item in self._np_include_dir_args:
                try:
                    self._np_include_dirs.append(item())  # The numpy callable
                except TypeError:
                    self._np_include_dirs.append(item)
        return self._np_include_dirs

    @include_dirs.setter
    def include_dirs(self, val):
        self._np_include_dir_args = val


# from MDAnalysis setup.py (https://www.mdanalysis.org/)
def get_numpy_include():
    try:
        # Obtain the numpy include directory. This logic works across numpy
        # versions.
        # setuptools forgets to unset numpy's setup flag and we get a crippled
        # version of it unless we do it ourselves.
        import builtins

        builtins.__NUMPY_SETUP__ = False
        import numpy as np
    except ImportError:
        try:
            # Try to install numpy
            from setuptools import dist

            dist.Distribution().fetch_build_eggs([REQUIRED_NUMPY])
            import numpy as np
        except Exception as e:
            print(e)
            print('*** package "numpy" not found ***')
            print("pyEDFlib requires a version of NumPy, even for setup.")
            print("Please get it from https://numpy.org/ or install it through your package manager.")
            sys.exit(-1)
    try:
        numpy_include = np.get_include()
    except AttributeError:
        numpy_include = np.get_numpy_include()
    return numpy_include





# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists("MANIFEST"):
    os.remove("MANIFEST")


if sys.platform == "darwin":
    # Don't create resource files on OS X tar.
    os.environ["COPY_EXTENDED_ATTRIBUTES_DISABLE"] = "true"
    os.environ["COPYFILE_DISABLE"] = "true"

make_ext_path = partial(os.path.join, "pyedflib", "_extensions")

if os.name == "nt":
    # Patch edflib.c
    with open(make_ext_path("c/edflib.c"), "rb") as fin, open(make_ext_path("c/edflib_utf8.c"), "wb") as fout:
        for line in fin:
            line = line.replace(b'#include "edflib.h"', b'#include "edflib.h"\r\n#include "fopen_utf8.h"')
            line = line.replace(b'file = fopeno(path, "rb");', b'file = fopen_utf8(path, "rb");')
            line = line.replace(b'file = fopeno(path, "wb");', b'file = fopen_utf8(path, "wb");')

            fout.write(line)

    sources = ["c/edflib_utf8.c", "c/fopen_utf8.c"]
    headers = ["c/edflib.h", "c/fopen_utf8.h"]
else:
    sources = ["c/edflib.c"]
    headers = ["c/edflib.h"]

sources = list(map(make_ext_path, sources))
headers = list(map(make_ext_path, headers))

cython_modules = ["_pyedflib"]
cython_sources = [("{0}.pyx" if USE_CYTHON else "{0}.c").format(module) for module in cython_modules]

c_macros = [("PY_EXTENSION", None), ("_LARGEFILE64_SOURCE", None), ("_LARGEFILE_SOURCE", None)]

cython_macros = []
cythonize_opts = {}
if os.environ.get("CYTHON_TRACE"):
    cythonize_opts["linetrace"] = True
    cython_macros.append(("CYTHON_TRACE_NOGIL", 1))

# By default C object files are rebuilt for every extension
# C files must be built once only for coverage to work
c_lib = (
    "c_edf",
    {
        "sources": sources,
        "depends": headers,
        "include_dirs": [make_ext_path("c"), sysconfig.get_path("include")],
        "macros": c_macros,
    },
)

ext_modules = [
    NumpyExtension(
        f"pyedflib._extensions.{module}",
        sources=[make_ext_path(source)],
        # Doesn't automatically rebuild if library changes
        depends=c_lib[1]["sources"] + c_lib[1]["depends"],
        include_dirs=[make_ext_path("c"), get_numpy_include()],
        define_macros=c_macros + cython_macros,
        libraries=[c_lib[0]],
    )
    for module, source in zip(cython_modules, cython_sources)
]


class develop_build_clib(develop):
    """Ugly monkeypatching to get clib to build for development installs
    See coverage comment above for why we don't just let libraries be built
    via extensions.
    All this is a copy of the relevant part of `install_for_development`
    for current master (Sep 2016) of setuptools.
    Note: if you want to build in-place with ``python setup.py build_ext``,
    that will only work if you first do ``python setup.py build_clib``.
    """

    def install_for_development(self):
        self.run_command("egg_info")

        # Build extensions in-place (the next 7 lines are the monkeypatch)
        import glob

        hitlist = glob.glob(os.path.join("build", "*", "c_edf.*"))
        if hitlist:
            # Remove existing clib - running build_clib twice in a row fails
            os.remove(hitlist[0])
        self.reinitialize_command("build_clib", inplace=1)
        self.run_command("build_clib")

        self.reinitialize_command("build_ext", inplace=1)
        self.run_command("build_ext")

        try:
            self.install_site_py()  # ensure that target dir is site-safe
        except Exception as e:
            print(e)

        if setuptools.bootstrap_install_from:
            self.easy_install(setuptools.bootstrap_install_from)
            setuptools.bootstrap_install_from = None

        # create an .egg-link in the installation dir, pointing to our egg
        from setuptools import log

        log.info("Creating %s (link to %s)", self.egg_link, self.egg_base)
        if not self.dry_run:
            with open(self.egg_link, "w") as f:
                f.write(f"{self.egg_path}\n{self.setup_path}")
        # postprocess the installed distro, fixing up .pth, installing scripts,
        # and handling requirements
        self.process_distribution(None, self.dist, not self.no_deps)


if __name__ == "__main__":
    if USE_CYTHON:
        ext_modules = cythonize(ext_modules, compiler_directives=cythonize_opts)

    setup(
        ext_modules=ext_modules,
        libraries=[c_lib],
        cmdclass={"develop": develop_build_clib},
    )
