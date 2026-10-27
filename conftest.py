"""Make the test suite import THIS checkout, not whatever `metacog` is installed in the venv.

Without this, a machine with the package installed (editable or not) from a different clone will
run these tests against that clone's code and report a green suite for the wrong tree. Prepending
here is enough because pytest imports `conftest.py` before collecting anything.
"""
import os
import sys

_REPO = os.path.dirname(os.path.abspath(__file__))
for _path in (os.path.join(_REPO, "src"), os.path.join(_REPO, "tracks", "industrial", "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
