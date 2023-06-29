import pytest
import pandas as pd

@pytest.fixture(scope="session")
def input_fpath():
    """fiture for getting the input file path
    """
    return "./example/data/data_processed_1prod_sample.json"

@pytest.fixture(scope="session")
def input_df(input_fpath):
    """fixture for creating the input dataframe
    """
    return pd.read_json(input_fpath, lines=True) 