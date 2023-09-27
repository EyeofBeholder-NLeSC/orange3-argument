"""Test the processor module"""
from contextlib import nullcontext as does_not_raise

import pandas as pd
import pytest
from pytest import approx

from orangearg.argument.miner.processor import (
    get_argument_topics,
    get_argument_sentiment,
    get_argument_coherence,
    update_argument_table,
    _match_list_size,
    _aggregate_list_by_another,
)
from orangearg.argument.miner.utilities import check_columns


@pytest.mark.parametrize(
    "dummy_list1, dummy_list2, exception_context",
    [
        ([1], [1], does_not_raise(enter_result=None)),
        ([1], [1, 2], pytest.raises(ValueError)),
    ],
)
def test__match_list_size(dummy_list1, dummy_list2, exception_context):
    """Unit test _match_list_size"""
    with exception_context as ex:
        _match_list_size(dummy_list1, dummy_list2)
    if ex:
        assert str(ex.value) == f"Input size not match: {(dummy_list1, dummy_list2)}."


def test__aggregate_list_by_another():
    """Unit test _aggregate_list_by_another."""
    dummy_keys = [1, 1, 2, 2]
    dummy_values = ["A", "B", "C", "D"]

    result = _aggregate_list_by_another(keys=dummy_keys, values=dummy_values)
    assert result == {1: ["A", "B"], 2: ["C", "D"]}


def test_get_argument_topics():
    """Unit test get_argument_topics."""
    dummy_ids = [0, 0, 1, 1, 1]
    dummy_topics = [-1, 0, 0, 1, 2]
    topics = get_argument_topics(arg_ids=dummy_ids, topics=dummy_topics)

    assert topics == [(-1, 0), (0, 1, 2)]


def test_get_argument_sentiment():
    """Unit test get_argument_sentiment."""
    dummy_ids = [0, 0, 1, 1, 1]
    dummy_ranks = [0.3, 0.7, 0.25, 0.2, 0.55]
    dummy_p_scores = [-0.5, 0.2, 0.3, 0.7, -0.45]
    sentiments = get_argument_sentiment(
        arg_ids=dummy_ids, ranks=dummy_ranks, p_scores=dummy_p_scores
    )

    assert sentiments == approx([0.495, 0.484], 0.01)


def test_get_argument_coherence():
    """Unit test get_argument_coherence"""
    dummy_sentiments = [-0.5, 0.8, 1, 0]
    dummy_scores = [1, 3, 2, 5]
    coherences = get_argument_coherence(
        sentiments=dummy_sentiments, scores=dummy_scores
    )

    assert coherences == approx([0.535, 0.799, 0.245, 0.082], 0.01)


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
