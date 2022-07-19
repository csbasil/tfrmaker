"""Handles tfrmaker imports for pytest."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src  # pylint: disable=unused-import, wrong-import-position
