"""Test the utilities module."""
import pytest
import pandas as pd

from orangearg.argument.miner.utilities import check_columns


def test_check_columns():
    """Unit test check_columns."""
    dummy_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    dummy_expected_cols = ["col1", "col3"]
    with pytest.raises(ValueError) as ex:
        check_columns(expected_cols=dummy_expected_cols, data=dummy_df)
    assert str(ex.value) == "Missing columns in the input dataframe: ['col3']."
