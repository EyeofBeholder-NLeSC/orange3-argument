"""Collection of helper functions."""
from typing import List

import pandas as pd


def check_columns(expected_cols: List[str], data: pd.DataFrame):
    """Check if a list of given columns exist in a given Pandas dataframe.

    Args:
        expected_cols (List[str]): list of columns to check
        df (pd.DataFrame): pandas dataframe to check

    Raises:
        ValueError: if any of the expected columns are missing.
    """
    missing_cols = [i for i in expected_cols if i not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in the input dataframe: {missing_cols}.")
