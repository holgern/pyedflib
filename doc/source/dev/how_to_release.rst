Guidelines for new releases for pyEDFlib
========================================

``vX.Y.Z`` refers to the release number.

Releases are built and published automatically by the ``Build wheel`` GitHub
Actions workflow (``.github/workflows/wheels.yml``). It runs on every pushed
tag matching ``vX.Y.Z``, builds the source distribution and the platform
wheels (Linux, macOS and Windows, CPython 3.8-3.14, including aarch64) and
uploads them to PyPI using the ``PYPI_API_TOKEN`` repository secret.

Bump the version
----------------

No manual version editing is needed. The version is automatically derived
from git tags by setuptools-scm. Simply create a signed tag for the release
(see below). Release versions (without ``.dev`` suffix) are created automatically
from annotated tags.

Add release notes
-----------------

Add a ``doc/release/X.Y.Z-notes.rst`` file describing the changes, together
with a matching ``doc/source/release.X.Y.Z.rst`` stub that includes it, and
link the new entry from ``doc/source/releasenotes.rst``.

Tag and push
------------

Create a signed tag and push it to GitHub::

    git tag -s vX.Y.Z -m "pyEDFlib X.Y.Z"
    git push origin vX.Y.Z

Pushing the tag triggers the ``Build wheel`` workflow, which builds the sdist
and wheels and publishes them to PyPI automatically. Follow the run on the
GitHub Actions tab and verify the new release appears on
https://pypi.org/project/pyEDFlib/.

Prepare for continued development
---------------------------------

No action needed. After creating a release tag, the next commit will
automatically have a development version (e.g., ``0.1.42.dev1+githash``)
when built with setuptools-scm.
