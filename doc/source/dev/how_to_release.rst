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

Set the version in ``setup.py`` by editing ``MAJOR``, ``MINOR`` and ``MICRO``,
and set ``ISRELEASED = True``. Commit the change.

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

Increment ``MICRO`` in ``setup.py`` and set ``ISRELEASED = False`` again, then
commit.
