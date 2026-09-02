"""
conftest.py (project root)
==========================

pytest automatically loads this file before running tests. We use it to make
sure the project root is on Python's import path, so "import app..." works
when running `pytest` from the project folder.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
