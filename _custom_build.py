"""
Custom PEP 517 build-backend wrapper for pyedflib.

This module wraps ``setuptools.build_meta`` and injects the C static
library (``c_edf``) and the Cython/C extension
(``pyedflib._extensions._pyedflib``) into every build.

Usage in ``pyproject.toml``::

    [build-system]
    requires = ["setuptools>=61", "numpy>=1.9.1", "cython>=0.29"]
    build-backend = "_custom_build"
    backend-path = ["."]

Because ``build-backend = "_custom_build"`` and ``backend-path = ["."]``,
pip / build will import *this file* as the backend. We forward all PEP 517
hooks to setuptools while adding our own extension/library definitions.

Tasks handled here (previously in setup.py):
  1. Version management via setuptools-scm (writes ``pyedflib/version.py`` at build time).
  2. Windows UTF-8 patch: generates ``edflib_utf8.c`` from ``edflib.c``.
  3. Static C library ``c_edf`` (edflib.c compiled once for coverage).
  4. Cython extension ``pyedflib._extensions._pyedflib`` with lazy
     numpy include-dir resolution.
"""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from functools import partial

import setuptools.build_meta as _stbm
import setuptools.dist

# ---------------------------------------------------------------------------
# Version reader — use setuptools-scm for git-based versioning
# ---------------------------------------------------------------------------


def _get_version() -> str:
    """
    Get version using setuptools-scm.
    
    Works from:
    - Git checkout: version derived from tags (e.g., 0.1.42 or 0.1.42.devN+githash)
    - Sdist: version read from embedded metadata
    - Fallback: returns 0.0.0.dev0 if all else fails
    """
    try:
        from setuptools_scm import get_version
        return get_version(root=".", relative_to=__file__)
    except (ImportError, LookupError):
        # setuptools-scm not available - should not happen during normal builds
        # but provide a safe fallback
        return "0.0.0.dev0"


# Get version from setuptools-scm
VERSION = _get_version()


# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------


def _git_revision() -> str:
    """Return the HEAD commit hash, or 'Unknown' on failure."""
    env = {k: os.environ[k] for k in ("SYSTEMROOT", "PATH") if k in os.environ}
    env.update({"LANGUAGE": "C", "LANG": "C", "LC_ALL": "C"})
    try:
        out = subprocess.Popen(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            env=env,
        ).communicate()[0]
        return out.strip().decode("ascii")
    except OSError:
        return "Unknown"


# ---------------------------------------------------------------------------
# Windows UTF-8 patch
# ---------------------------------------------------------------------------

_make_ext_path = partial(os.path.join, "pyedflib", "_extensions")


def _patch_edflib_utf8() -> None:
    """On Windows, create ``edflib_utf8.c`` with UTF-8-aware fopen calls."""
    src = _make_ext_path("c/edflib.c")
    dst = _make_ext_path("c/edflib_utf8.c")
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        for line in fin:
            line = line.replace(
                b'#include "edflib.h"',
                b'#include "edflib.h"\r\n#include "fopen_utf8.h"',
            )
            line = line.replace(
                b'file = fopeno(path, "rb");',
                b'file = fopen_utf8(path, "rb");',
            )
            line = line.replace(
                b'file = fopeno(path, "wb");',
                b'file = fopen_utf8(path, "wb");',
            )
            fout.write(line)


# ---------------------------------------------------------------------------
# Extension / library definitions
# ---------------------------------------------------------------------------


def _get_numpy_include() -> str:
    """Return NumPy's include directory."""
    try:
        import builtins

        builtins.__NUMPY_SETUP__ = False  # type: ignore[attr-defined]
        import numpy as np
    except ImportError:
        print("*** package 'numpy' not found — cannot build extension ***")
        sys.exit(1)
    try:
        return np.get_include()
    except AttributeError:
        return np.get_numpy_include()  # type: ignore[attr-defined]


def get_libraries():
    """Return list of (name, build-info) tuples for static C libraries."""
    if os.name == "nt":
        _patch_edflib_utf8()
        sources = ["c/edflib_utf8.c", "c/fopen_utf8.c"]
        headers = ["c/edflib.h", "c/fopen_utf8.h"]
    else:
        sources = ["c/edflib.c"]
        headers = ["c/edflib.h"]

    sources = list(map(_make_ext_path, sources))
    headers = list(map(_make_ext_path, headers))

    c_macros = [
        ("PY_EXTENSION", None),
        ("_LARGEFILE64_SOURCE", None),
        ("_LARGEFILE_SOURCE", None),
    ]

    return [
        (
            "c_edf",
            {
                "sources": sources,
                "depends": headers,
                "include_dirs": [_make_ext_path("c"), sysconfig.get_path("include")],
                "macros": c_macros,
            },
        )
    ]


def get_ext_modules():
    """Return the list of Extension objects (Cython or pre-built C)."""
    try:
        from Cython.Build import cythonize

        use_cython = True
    except ImportError as e:
        use_cython = False
        c_path = _make_ext_path("_pyedflib.c")
        if not os.path.exists(c_path):
            raise RuntimeError(
                "Cython must be installed when working with a development "
                "version of PyEDFlib (no pre-built _pyedflib.c found)"
            ) from e

    c_macros = [
        ("PY_EXTENSION", None),
        ("_LARGEFILE64_SOURCE", None),
        ("_LARGEFILE_SOURCE", None),
    ]
    cython_macros: list = []
    cythonize_opts: dict = {}
    if os.environ.get("CYTHON_TRACE"):
        cythonize_opts["linetrace"] = True
        cython_macros.append(("CYTHON_TRACE_NOGIL", 1))

    libs = get_libraries()
    c_lib_name = libs[0][0]
    c_lib_sources = libs[0][1]["sources"]
    c_lib_headers = libs[0][1]["depends"]

    source_file = "_pyedflib.pyx" if use_cython else "_pyedflib.c"

    from setuptools import Extension

    # We need numpy include dirs but numpy may not be importable yet when
    # setuptools first resolves extensions.  Work around with a lazy subclass.
    class _LazyNumpyExtension(Extension):
        """Extension that defers numpy include resolution until compile time."""

        _np_include_resolved: bool = False

        def _resolve_np(self):
            if not self._np_include_resolved:
                self._np_include_resolved = True
                self._include_dirs.append(_get_numpy_include())

        @property
        def include_dirs(self):  # type: ignore[override]
            self._resolve_np()
            return self._include_dirs

        @include_dirs.setter
        def include_dirs(self, val):
            self._include_dirs = list(val)

    ext = _LazyNumpyExtension(
        "pyedflib._extensions._pyedflib",
        sources=[_make_ext_path(source_file)],
        depends=c_lib_sources + c_lib_headers,
        include_dirs=[_make_ext_path("c")],
        define_macros=c_macros + cython_macros,
        libraries=[c_lib_name],
    )

    if use_cython:
        from Cython.Build import cythonize

        exts = cythonize([ext], compiler_directives=cythonize_opts)
    else:
        exts = [ext]

    return exts


# ---------------------------------------------------------------------------
# PEP 517 build-backend wrapper
# ---------------------------------------------------------------------------
# Import everything from setuptools.build_meta so we expose all the
# required PEP 517/660 hooks without reimplementing them.
# ---------------------------------------------------------------------------

# Capture the original finalize_options once at import time to avoid
# stacking monkey-patches across multiple hook calls.
_orig_finalize = setuptools.dist.Distribution.finalize_options


def _patched_finalize(self, *args, **kwargs):
    """Patched finalize_options that injects our libraries and extensions."""
    _orig_finalize(self, *args, **kwargs)
    # Avoid double-injection on repeated calls
    if getattr(self, "_pyedflib_injected", False):
        return
    self._pyedflib_injected = True
    self.libraries = list(self.libraries or []) + get_libraries()
    self.ext_modules = list(self.ext_modules or []) + get_ext_modules()


# Apply the monkey-patch once at import time
setuptools.dist.Distribution.finalize_options = _patched_finalize


def _prepare_build():
    """Run once before any build hook to write version file via setuptools-scm."""
    try:
        from setuptools_scm import dump_version
        dump_version(root=".", version=VERSION, write_to="pyedflib/version.py")
    except ImportError:
        # setuptools-scm not available, skip version file generation
        # (should not happen during normal builds)
        pass


# Override each PEP 517 hook to call _prepare_build() first.

def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):  # type: ignore[no-redef]
    _prepare_build()
    return _stbm.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):  # type: ignore[no-redef]
    _prepare_build()
    return _stbm.build_sdist(sdist_directory, config_settings)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):  # type: ignore[no-redef]
    _prepare_build()
    return _stbm.build_editable(wheel_directory, config_settings, metadata_directory)


def get_requires_for_build_wheel(config_settings=None):  # type: ignore[no-redef]
    return _stbm.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_sdist(config_settings=None):  # type: ignore[no-redef]
    return _stbm.get_requires_for_build_sdist(config_settings)


def get_requires_for_build_editable(config_settings=None):  # type: ignore[no-redef]
    return _stbm.get_requires_for_build_editable(config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):  # type: ignore[no-redef]
    _prepare_build()
    return _stbm.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):  # type: ignore[no-redef]
    _prepare_build()
    return _stbm.prepare_metadata_for_build_editable(
        metadata_directory, config_settings
    )
