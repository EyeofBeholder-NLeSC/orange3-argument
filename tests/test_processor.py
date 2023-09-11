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
    _match_list_size,
    _aggregate_list_by_another,
)
from .conftest import (
    mock_processor__aggregate_list_by_another,
    mock_processor__match_list_size,
)


def test_check_columns():
    """Unit test check_columns."""
    dummy_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    dummy_expected_cols = ["col1", "col3"]
    with pytest.raises(AssertionError) as ex:
        check_columns(expected_cols=dummy_expected_cols, data=dummy_df)
    assert str(ex.value) == "Missing columns in the input dataframe: ['col3']."


def test__match_list_size():
    """Unit test _match_list_size"""
    dummy_list1 = [1, 2, 3]
    dummy_list2 = [2, 3, 4]

    _match_list_size(dummy_list1, dummy_list2)


def test__match_list_size_negative():
    """Negative test _match_list_size."""
    dummy_list1 = [1, 2, 3]
    dummy_list2 = [1, 2, 3, 4]

    with pytest.raises(AssertionError) as ex:
        _match_list_size(dummy_list1, dummy_list2)
    assert str(ex.value) == "Input size not match: [3, 4]."


def test__aggregate_list_by_another():
    """Unit test _aggregate_list_by_another."""
    dummy_keys = [1, 1, 2, 2]
    dummy_values = ["A", "B", "C", "D"]

    result = _aggregate_list_by_another(keys=dummy_keys, values=dummy_values)
    assert result == {1: ["A", "B"], 2: ["C", "D"]}


def test_get_argument_topics(
    mock_processor__match_list_size,
    mock_processor__aggregate_list_by_another,
):
    """Unit test get_argument_topics."""
    topics = get_argument_topics(arg_ids=[], topics=[])

    assert topics == [[0.4, 0.6], [0.5, 0.5]]
    mock_processor__match_list_size.assert_called_once()
    mock_processor__aggregate_list_by_another.assert_called_once_with(
        keys=[], values=[]
    )


def test_get_argument_sentiment(
    mock_processor__match_list_size,
    mock_processor__aggregate_list_by_another,
):
    """Unit test get_argument_sentiment."""
    sentiments = get_argument_sentiment(arg_ids=[], ranks=[], p_scores=[])

    assert sentiments == [0.76, 0.75]
    mock_processor__match_list_size.assert_called_once()
    mock_processor__aggregate_list_by_another.assert_called_with(keys=[], values=[])


def test_get_argument_coherence(mock_processor__match_list_size):
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
    mock_processor__match_list_size.assert_called_once()


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


def test_integrate_processor(df_arguments, df_chunks):
    """Integrate test the processor module."""
    expected_cols_chunks = ["argument_id", "topic", "rank", "polarity_score"]
    expected_cols_arguments = ["score"]

    check_columns(expected_cols=expected_cols_chunks, data=df_chunks)
    check_columns(expected_cols=expected_cols_arguments, data=df_arguments)

    chunk_arg_ids = df_chunks["argument_id"]
    chunk_topics = df_chunks["topic"]
    chunk_ranks = df_chunks["rank"]
    chunk_p_scores = df_chunks["polarity_score"]
    arg_scores = df_arguments["score"]

    arg_topics = get_argument_topics(arg_ids=chunk_arg_ids, topics=chunk_topics)
    arg_sentiments = get_argument_sentiment(
        arg_ids=chunk_arg_ids, ranks=chunk_ranks, p_scores=chunk_p_scores
    )
    arg_coherences = get_argument_coherence(
        scores=arg_scores, sentiments=arg_sentiments
    )
    assert all([0.0 <= s <= 1.0 for s in arg_sentiments])
    assert all([0.0 <= c <= 1.0 for c in arg_coherences])
    assert (
        len(arg_topics)
        == len(arg_sentiments)
        == len(arg_coherences)
        == df_arguments.shape[0]
    )

    df_arguments_updated = update_argument_table(
        df_arguments=df_arguments,
        topics=arg_topics,
        sentiments=arg_sentiments,
        coherences=arg_coherences,
    )
