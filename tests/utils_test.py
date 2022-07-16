"""Utility functions for making data driven testing."""

import pathlib
import json


def inject_test_data(filename):
    """
    Read the content of the JSON file and convert it to
    a named tuple, can be used for injecting test data set to tests,
    helps in separating test data from the tests
    """
    file = pathlib.Path(filename)
    with open(file, encoding="utf8") as file_obj:
        data = json.load(file_obj)
    return data
