"""Test the processor module"""
import pandas as pd
import pytest
from pytest import approx

from orangecontrib.argument.miner.processor import (
    check_columns,
    get_argument_topics,
    get_argument_sentiment,
    get_argument_coherence,
    update_argument_table,
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


def test_get_argument_coherence():
    """Unit test get_argument_coherence"""
    dummy_sentiments = [-0.5, 0.8, 1, 0]
    dummy_scores = [1, 3, 2, 5]
    coherences = get_argument_coherence(
        sentiments=dummy_sentiments, scores=dummy_scores
    )

    assert coherences[0] == approx(0.535, 0.01)
    assert coherences[1] == approx(0.799, 0.01)
    assert coherences[2] == approx(0.245, 0.01)
    assert coherences[3] == approx(0.082, 0.01)


def test_get_argument_coherence_input_size_mismatch():
    """Edge case get_argument_coherence: input size mismatch."""
    dummy_sentiments = [-0.5, 0.8, 1, 0, 0.2]
    dummy_scores = [1, 3, 2, 5]
    with pytest.raises(AssertionError) as ex:
        get_argument_coherence(sentiments=dummy_sentiments, scores=dummy_scores)
    assert str(ex.value) == "Size of scores and sentiments not match!"


def test_update_argument_table(mocker):
    """Unit test update_argument_table"""
    dummy_topics = [[1, 2], [3, 4]]
    dummy_sentiments = [0.4, 0.8]
    dummy_coherences = [0.1, 0.3]
    dummy_df_arguments = pd.DataFrame()
    expected_result = pd.DataFrame(
        {
            "topics": dummy_topics,
            "sentiment": dummy_sentiments,
            "coherence": dummy_coherences,
        }
    )
    mock_deepcopy = mocker.patch("copy.deepcopy", return_value=dummy_df_arguments)

    result = update_argument_table(
        df_arguments=dummy_df_arguments,
        topics=dummy_topics,
        sentiments=dummy_sentiments,
        coherences=dummy_coherences,
    )

    assert result.equals(expected_result)
    mock_deepcopy.assert_called_once_with(dummy_df_arguments)


@pytest.mark.skip(reason="To be added.")
def test_integrate_processor():
    pass
