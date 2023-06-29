import pandas as pd
import pytest
from reader import read_json_file, validate

def test_read_json_file_existing_file(input_fpath):
    # Test reading an existing JSON file
    result = read_json_file(input_fpath)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (371, 3)  # Check the number of rows in the DataFrame

def test_read_json_file_non_existing_file():
    # Test reading a non-existing JSON file
    fpath = "non_existing_file.json"
    with pytest.raises(AssertionError):
        read_json_file(fpath)

def test_validate_correct_columns():
    # Test validating input DataFrame with correct columns
    data = {
        "argument": ["A", "B", "C"],
        "score": [1, 2, 3]
    }
    df_arguments = pd.DataFrame(data)
    result = validate(df_arguments)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3  # Check the number of rows in the DataFrame

def test_validate_missing_argument_column():
    # Test validating input DataFrame with missing argument text column
    data = {
        "score": [1, 2, 3]
    }
    df_arguments = pd.DataFrame(data)
    with pytest.raises(AssertionError):
        validate(df_arguments)

def test_validate_missing_score_column():
    # Test validating input DataFrame with missing score text column
    data = {
        "argument": ["A", "B", "C"]
    }
    df_arguments = pd.DataFrame(data)
    with pytest.raises(AssertionError):
        validate(df_arguments)
