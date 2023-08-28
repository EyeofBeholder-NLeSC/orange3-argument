"""Argument filre reader module. 

This module provides functions to read intput data in different formats. 
For now, only JSON format is supported, though it's possible to support 
other formats in the future.
"""

import os
import json
from json import JSONDecodeError

import pandas as pd


def read_json_file(fpath: str):
    """Read json file into flat table.

    Things handled include:
        - multiple json objects
        - semi-structured json string
    """

    assert os.path.isfile(fpath)

    with open(fpath, "r", encoding="utf-8") as json_file:
        try:
            json_data = json.loads(json_file.read())
        except JSONDecodeError as error:
            json_file.seek(0)  # go back to start of file

            # multiple json objects in one file
            if "Extra data" in str(error):
                json_data = []
                for json_object in json_file:
                    json_data.append(json.loads(json_object))

    # read semi-structured json as flat table
    return pd.json_normalize(json_data)
