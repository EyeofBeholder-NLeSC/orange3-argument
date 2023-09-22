"""Tests of the reader module"""
import pytest
import pandas as pd

from orangearg.argument.miner.reader import read_json_file


@pytest.fixture
def mocker_multi_objs(mocker):
    """Mock-open a file containing multiple JSON objects"""
    mocked_data = mocker.mock_open(
        read_data='{"col1": 1, "col2": 2}\n{"col1": 3, "col2": 4}\n'
    )
    mocker.patch("builtins.open", mocked_data)


@pytest.fixture
def mocker_semi_struct(mocker):
    """Mock-open a semi-structured JSON file"""
    mocked_data = mocker.mock_open(
        read_data='{"col1": 1, "col2": {"subcol1": 2, "subcol2": 3}}'
    )
    mocker.patch("builtins.open", mocked_data)


class TestReadJSONFile:
    """Tests of the read_json_file function"""

    def test_multi_object(self, mocker_multi_objs):
        """Test if the function can read a JSON file with multiple objects correctly."""
        data = read_json_file("fakefile")
        expected_data = pd.DataFrame({"col1": [1, 3], "col2": [2, 4]})
        assert data.equals(expected_data)

    def test_semi_struct(self, mocker_semi_struct):
        """Test if the function can read a semi-structure JSON file well"""
        data = read_json_file("fakefile")
        expected_data = pd.DataFrame(
            {"col1": [1], "col2.subcol1": [2], "col2.subcol2": [3]}
        )
        assert data.equals(expected_data)
