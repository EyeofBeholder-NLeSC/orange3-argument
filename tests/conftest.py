import pytest
import pandas as pd

@pytest.fixture(scope="session")
def input_fpath():
    """fiture for getting the input file path
    """
    yield "./example/data/data_processed_1prod_full.json"

@pytest.fixture(scope="session")
def input_df(input_fpath):
    """fixture for creating the input dataframe
    """
    df = pd.read_json(input_fpath, lines=True)
    df = df.rename(columns={
        "reviewText": "argument", 
        "overall": "score"
    })
    df = df.drop(columns=["vote"]) 
    df["argument"] = df["argument"].astype(str)
    yield df