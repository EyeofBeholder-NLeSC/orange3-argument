"""Test the processor module"""
import pandas as pd
import pytest

from orangecontrib.argument.miner.processor import get_argument_topics


def test_get_argument_topics():
    """Unit test get_argument_topics."""
    dummy_input = pd.DataFrame({"argument_id": [0, 0, 1, 1], "topic": [1, 2, 2, 3]})
    topics = get_argument_topics(dummy_input)

    assert topics == [[1, 2], [2, 3]]


def test_get_argument_topics_missing_cols():
    """Edge case get_argument_topics: required columns missing."""
    dummy_input = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    with pytest.raises(AssertionError) as e:
        get_argument_topics(dummy_input)
    assert "Missing required columns in df_chunks: argument_id, topic." in str(e.value)
