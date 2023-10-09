#!/usr/bin/env python

import os
import sys
import subprocess
import setuptools
from functools import partial

from setuptools import setup, Extension
from distutils.sysconfig import get_python_inc

try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False
    if not os.path.exists(os.path.join('pyedflib', '_extensions', '_pyedflib.c')):
        msg = ("Cython must be installed when working with a development "
               "version of PyEDFlib")
        raise RuntimeError(msg)


MAJOR = 0
MINOR = 1
MICRO = 36
ISRELEASED = True
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

# Version of Numpy required for setup
REQUIRED_NUMPY = 'numpy>=1.9.1'


# from MDAnalysis setup.py (https://www.mdanalysis.org/)
class NumpyExtension(Extension):
    """Derived class to cleanly handle setup-time (numpy) dependencies.
    """
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
        try:
            import __builtin__  # py2
            __builtin__.__NUMPY_SETUP__ = False
        except:
            import builtins  # py3
            builtins.__NUMPY_SETUP__ = False
        import numpy as np
    except ImportError as e:
        try:
            # Try to install numpy
            from setuptools import dist
            dist.Distribution().fetch_build_eggs([REQUIRED_NUMPY])
            import numpy as np
        except Exception as e:
            print(e)
            print('*** package "numpy" not found ***')
            print('pyEDFlib requires a version of NumPy, even for setup.')
            print('Please get it from https://numpy.org/ or install it through '
                  'your package manager.')
            sys.exit(-1)
    try:
        numpy_include = np.get_include()
    except AttributeError:
        numpy_include = np.get_numpy_include()
    return numpy_include

# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION


def get_version_info():
    # Adding the git rev number needs to be done inside
    # write_version_py(), otherwise the import of pyedflib.version messes
    # up the build under Python 3.
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    elif os.path.exists('pyedflib/version.py'):
        # must be a source distribution, use existing version file
        # load it as a separate module to not load pyedflib/__init__.py
        import types
        from importlib.machinery import SourceFileLoader
        loader = SourceFileLoader('pyedflib.version', 'pyedflib/version.py')
        version = types.ModuleType(loader.name)
        loader.exec_module(version)
        GIT_REVISION = version.git_revision
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += '.dev0+' + GIT_REVISION[:7]

    return FULLVERSION, GIT_REVISION


def write_version_py(filename='pyedflib/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM pyedflib SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s

if not release:
    version = full_version
"""
    FULLVERSION, GIT_REVISION = get_version_info()

    with open(filename, 'w') as a:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION,
                       'git_revision': GIT_REVISION,
                       'isrelease': str(ISRELEASED)})


# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')


if sys.platform == "darwin":
    # Don't create resource files on OS X tar.
    os.environ["COPY_EXTENDED_ATTRIBUTES_DISABLE"] = "true"
    os.environ["COPYFILE_DISABLE"] = "true"

make_ext_path = partial(os.path.join, "pyedflib", "_extensions")

sources = ["c/edflib.c"]
sources = list(map(make_ext_path, sources))
headers = ["c/edflib.h"]
headers = list(map(make_ext_path, headers))

cython_modules = ['_pyedflib']
cython_sources = [('{0}.pyx' if USE_CYTHON else '{0}.c').format(module)
                  for module in cython_modules]

c_macros = [("PY_EXTENSION", None), ("_LARGEFILE64_SOURCE", None), ("_LARGEFILE_SOURCE", None)]

cython_macros = []
cythonize_opts = {}
if os.environ.get("CYTHON_TRACE"):
    cythonize_opts['linetrace'] = True
    cython_macros.append(("CYTHON_TRACE_NOGIL", 1))

# By default C object files are rebuilt for every extension
# C files must be built once only for coverage to work
c_lib = ('c_edf',{'sources': sources,
                 'depends': headers,
                 'include_dirs': [make_ext_path("c"), get_python_inc()],
                 'macros': c_macros,})

ext_modules = [
    NumpyExtension(f'pyedflib._extensions.{module}',
              sources=[make_ext_path(source)],
              # Doesn't automatically rebuild if library changes
              depends=c_lib[1]['sources'] + c_lib[1]['depends'],
              include_dirs=[make_ext_path("c"), get_numpy_include()],
              define_macros=c_macros + cython_macros,
              libraries=[c_lib[0]])
    for module, source, in zip(cython_modules, cython_sources)
]

from setuptools.command.develop import develop
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
        self.run_command('egg_info')

        # Build extensions in-place (the next 7 lines are the monkeypatch)
        import glob
        hitlist = glob.glob(os.path.join('build', '*', 'c_edf.*'))
        if hitlist:
            # Remove existing clib - running build_clib twice in a row fails
            os.remove(hitlist[0])
        self.reinitialize_command('build_clib', inplace=1)
        self.run_command('build_clib')

        self.reinitialize_command('build_ext', inplace=1)
        self.run_command('build_ext')

        try:
            self.install_site_py()  # ensure that target dir is site-safe
        except Exception as e:
            print(e)

        if setuptools.bootstrap_install_from:
            self.easy_install(setuptools.bootstrap_install_from)
            setuptools.bootstrap_install_from = None

        # create an .egg-link in the installation dir, pointing to our egg
        from distutils import log
        log.info("Creating %s (link to %s)", self.egg_link, self.egg_base)
        if not self.dry_run:
            with open(self.egg_link, "w") as f:
                f.write(self.egg_path + "\n" + self.setup_path)
        # postprocess the installed distro, fixing up .pth, installing scripts,
        # and handling requirements
        self.process_distribution(None, self.dist, not self.no_deps)


if __name__ == '__main__':

    # Rewrite the version file every time
    write_version_py()
    if USE_CYTHON:
            ext_modules = cythonize(ext_modules, compiler_directives=cythonize_opts)

    setup(
        name="pyEDFlib",
        maintainer="Holger Nahrstaedt",
        maintainer_email="nahrstaedt@gmail.com",
        author='Holger Nahrstaedt',
        author_email='nahrstaedt@gmail.com',
        url="https://github.com/holgern/pyedflib",
        license="BSD",
        description="library to read/write EDF+/BDF+ files",
        long_description=open('README.rst').read(),
        keywords=["EDFlib", "European data format", "EDF", "BDF", "EDF+", "BDF+"],
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Operating System :: OS Independent",
            "Programming Language :: C",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Topic :: Software Development :: Libraries :: Python Modules"
        ],
        platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
        version=get_version_info()[0],
        packages=['pyedflib','pyedflib._extensions','pyedflib.data', 'pyedflib.tests', 'pyedflib.tests.data'],
        package_data={'pyedflib.data': ['*.edf', '*.bdf'], 'pyedflib.tests.data': ['*.edf', '*.bdf'], },
        ext_modules=ext_modules,
        libraries=[c_lib],
        cmdclass={'develop': develop_build_clib},
        install_requires=[REQUIRED_NUMPY],
    )
