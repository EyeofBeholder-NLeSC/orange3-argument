"""Argument filre reader module. 

For now, only JSON format is supported, though it's possible
to support other formats in the future.
"""

import pandas as pd
import os
import json
from json import JSONDecodeError


def read_json_file(fpath:str):
    """Read json file into flat table.
   
    Things handled include:
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

def validate(df_arguments):
    """Validate input argument dataframe.
    
    Things handled include:
        - check number and dtype of columns
        - rename columns for future analysis
    """
    assert df_arguments.columns.size == 2, \
        "More columns found, only need argument texts and scores!"
        
    argument_col = df_arguments.select_dtypes(include=["object"]).columns
    score_col = df_arguments.select_dtypes(include=["number"]).columns
    assert argument_col.size == 1, "Missing argument text column!"
    assert score_col.size == 1, "Missing score text column!"
    
    mapper = {
        argument_col[0]: "argument", 
        score_col[0]: "score"
    }
    df_arguments = df_arguments.rename(columns=mapper)
    df_arguments["argument"] = df_arguments["argument"].astype(str) # to string type
    
    return df_arguments