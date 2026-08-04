# Get pyEDFlib version information from package metadata (via setuptools-scm)

from importlib.metadata import version as _get_version

__version__ = _get_version("pyEDFlib")
version = __version__
short_version = __version__
full_version = __version__
git_revision = "Unknown"
release = True
