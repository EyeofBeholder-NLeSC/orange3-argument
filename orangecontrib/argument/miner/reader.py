"""Argument filre reader module

This module implements functions for reading input data files in
different formats. So far, we only have the support to JSON file.
But we forsee the need of supporting other formats, and all future
functions in this scope should be in this module.
"""

import json

import pandas as pd


def read_json_file(fpath: str) -> pd.DataFrame:
    """Read a local JSON file and return its content as a pandas dataframe.

    This function will automatically handle the case that a JSON
    file contains multiple JSON objects. It will also normalize
    semi-structured JSON strings.

    Args:
        fpath (str): The file path

    Returns:
        pd.DataFrame: The pandas dataframe object that contains
        content of the JSON file read from the given path.

    """

    with open(fpath, "r", encoding="utf-8") as json_file:
        json_data = []
        # handle multiple JSON objects in one file
        for json_object in json_file:
            json_data.append(json.loads(json_object))

    # normalize semi-structured json as flat table
    return pd.json_normalize(json_data)
