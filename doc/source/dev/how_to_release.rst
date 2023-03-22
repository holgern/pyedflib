Guidelines for new releases for pyedflib
========================================
vX.X.X refers to the release number

Tag the release and trigger building of wheels in appvoyer
----------------------------------------------------------
Change ISRELEASED in setup.py to True and commit.

Appveyor will now build wheels for windows.

Tag the release with

```git tag -s vX.X.X```

and push the tag to master.

Clean up source
---------------
Remove untraced files with git clean

First check which files will be deleted:

```git clean -xfdn```

Then run without -n:

```git clean -xfd```

Create the source distribution files with:

```python setup.py sdist --formats=gztar,zip```

Upload the release and windows wheels to pypi
---------------------------------------------

Download all wheels from Appveyor and put them into the dist directory

Register all files with

```twine register dist\filename_which_should_uploaded.whl```

and upload with

```twine upload  dist\filename_which_should_uploaded.whl```

Prepare for continued development
---------------------------------

Increment the version number in setup.py and change ISRELEASED to False.
