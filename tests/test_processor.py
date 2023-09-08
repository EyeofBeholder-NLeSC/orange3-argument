"""Test the processor module"""
from unittest import mock

import pandas as pd
import pytest

from orangecontrib.argument.miner.processor import (
    check_columns,
    get_argument_topics,
    get_argument_sentiment,
)
from .conftest import mock_check_columns


def test_check_columns():
    """Unit test check_columns."""
    dummy_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    dummy_expected_cols = ["col1", "col3"]
    with pytest.raises(AssertionError) as ex:
        check_columns(expected_cols=dummy_expected_cols, df=dummy_df)
    assert str(ex.value) == "Missing required columns in df: col3"


def test_get_argument_topics(mock_check_columns):
    """Unit test get_argument_topics."""
    dummy_input = pd.DataFrame(
        {
            "argument_id": [0, 0, 1, 1],
            "topic": [1, 2, 2, 3],
        }
    )
    topics = get_argument_topics(dummy_input)

    assert topics == [[1, 2], [2, 3]]
    mock_check_columns.assert_called_once()


def test_get_argument_sentiment(mock_check_columns):
    """Unit test get_argument_sentiment."""
    dummy_input = pd.DataFrame(
        {
            "argument_id": [0, 0, 1, 1],
            "rank": [0.5, 0.5, 0.4, 0.6],
            "polarity_score": [-1, 0, 0.5, -0.4],
        }
    )
    sentiments = get_argument_sentiment(dummy_input)

    assert sentiments == [0.25, 0.48]
    mock_check_columns.assert_called_once()
