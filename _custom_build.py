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
  1. Version constants and ``write_version_py()`` (regenerates
     ``pyedflib/version.py`` at build time).
  2. Windows UTF-8 patch: generates ``edflib_utf8.c`` from ``edflib.c``.
  3. Static C library ``c_edf`` (edflib.c compiled once for coverage).
  4. Cython extension ``pyedflib._extensions._pyedflib`` with lazy
     numpy include-dir resolution.
"""

from __future__ import annotations

import re
import os
import subprocess
import sys
import sysconfig
from functools import partial

import setuptools.dist

# ---------------------------------------------------------------------------
# Version constants — read from pyproject.toml (single source of truth)
# ---------------------------------------------------------------------------

def _get_version_from_pyproject() -> str:
    """Read version from pyproject.toml."""
    with open("pyproject.toml") as f:
        content = f.read()
    # Find version in [project] section: version = "0.1.33"
    match = re.search(r'^\[project\]\s*$.*?^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE | re.DOTALL)
    if match:
        return match.group(1)
    raise RuntimeError("Could not find version in pyproject.toml")


VERSION = _get_version_from_pyproject()
ISRELEASED = True


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


def get_version_info() -> tuple[str, str]:
    """Return (full_version, git_revision)."""
    full_version = VERSION
    if os.path.exists(".git"):
        git_rev = _git_revision()
    elif os.path.exists("pyedflib/version.py"):
        import types
        from importlib.machinery import SourceFileLoader

        loader = SourceFileLoader("pyedflib.version", "pyedflib/version.py")
        mod = types.ModuleType(loader.name)
        loader.exec_module(mod)
        git_rev = mod.git_revision  # type: ignore[attr-defined]
    else:
        git_rev = "Unknown"

    if not ISRELEASED:
        full_version += f".dev0+{git_rev[:7]}"

    return full_version, git_rev


def write_version_py(filename: str = "pyedflib/version.py") -> None:
    """Regenerate ``pyedflib/version.py``."""
    template = (
        "# THIS FILE IS GENERATED FROM pyedflib build.py\n"
        "short_version = '{version}'\n"
        "version = '{version}'\n"
        "full_version = '{full_version}'\n"
        "git_revision = '{git_revision}'\n"
        "release = {isrelease}\n"
        "\n"
        "if not release:\n"
        "    version = full_version\n"
    )
    full_version, git_rev = get_version_info()
    with open(filename, "w") as fh:
        fh.write(
            template.format(
                version=VERSION,
                full_version=full_version,
                git_revision=git_rev,
                isrelease=str(ISRELEASED),
            )
        )


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
    except ImportError:
        use_cython = False
        c_path = _make_ext_path("_pyedflib.c")
        if not os.path.exists(c_path):
            raise RuntimeError(
                "Cython must be installed when working with a development "
                "version of PyEDFlib (no pre-built _pyedflib.c found)"
            )

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

# Monkey-patch setuptools.dist.Distribution so that our libraries and
# ext_modules are merged in before the build starts.

_orig_finalize = setuptools.dist.Distribution.finalize_options


def _patched_finalize(self, *args, **kwargs):
    _orig_finalize(self, *args, **kwargs)
    # Avoid double-injection on repeated calls
    if getattr(self, "_pyedflib_injected", False):
        return
    self._pyedflib_injected = True
    self.libraries = list(self.libraries or []) + get_libraries()
    self.ext_modules = list(self.ext_modules or []) + get_ext_modules()


setuptools.dist.Distribution.finalize_options = _patched_finalize


def _prepare_build():
    """Run once before any build hook to inject extensions and version file."""
    write_version_py()

# Override each PEP 517 hook to call _prepare_build() first.

import setuptools.build_meta as _stbm  # noqa: E402


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
    return _stbm.prepare_metadata_for_build_editable(metadata_directory, config_settings)
