"""Argument filre reader module. 

For now, only JSON format is supported, though it's possible
to support other formats in the future.
"""

import pandas as pd
import os
import json
from json import JSONDecodeError


def read_json_file(fpath:str):
    """Read json file while handling validation.
   
    Cases to be handled include:
        - multiple json objects
        - semi-structured json string 
    """
        
    assert os.path.isfile(fpath), "File not exist: %s." % fpath
   
    with open(fpath, "r") as json_file:
        try:
            json_data = json.loads(json_file.read()) 
        except JSONDecodeError as e:
            json_file.seek(0) # go back to start of file
            if "Extra data" in str(e): # multiple json objects in one file
                json_data = []
                for json_object in json_file:
                    json_data.append(json.loads(json_object))
    
    return pd.json_normalize(json_data) # read semi-structured json as flat table

