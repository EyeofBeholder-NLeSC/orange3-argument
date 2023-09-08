"""Argument processor module."""

from typing import List
import copy
import math

import numpy as np
import pandas as pd


def check_columns(expected_cols: List[str], df: pd.DataFrame):
    """Check if a list of given columns exist in a given Pandas dataframe.

    Args:
        expected_cols (List[str]): list of columns to check
        df (pd.DataFrame): pandas dataframe to check
    """
    assert set(expected_cols).issubset(
        set(df.columns)
    ), "Missing required columns in df: {missing_cols}".format(
        missing_cols=", ".join([i for i in expected_cols if i not in df.columns])
    )


def get_argument_topics(df_chunks: pd.DataFrame) -> List[list[int]]:
    """Get argument topics.

    The topics of an argument is a combination of the topics of all chunks that belong to this argument. Duplications are not removed, and the reason behind is that duplications can be treated as a sign of topic importance. Also, even though two chunks can belong to the same topic, they could still have different ranks within an argument.

    Args:
        df_chunks (pd.DataFrame): Table of chunks, which should contain at least two columns that are `argument_id` and `topic`, where `argument_id` keeps the id of the argument a chunk belongs to, and `topic` stores the topic index of a chunk.

    Returns:
        List[list[int]]: list of argument topics, which is also a list containing topic indices of chunks belonging to this argument.
    """
    expected_cols = ["argument_id", "topic"]
    check_columns(expected_cols=expected_cols, df=df_chunks)

    topics = df_chunks.groupby(
        by="argument_id",
        as_index=False,
    ).agg(
        {"topic": list}
    )["topic"]

    return list(topics)


def get_argument_sentiment(df_chunks: pd.DataFrame) -> List[float]:
    """Get argument sentiment score.

    The sentiment score of an argument is calculated as a weighted sum of sentiment scores of chunks belonging to this argument, where weights are ranks of the chunks. The result score is then normalized into range [0, 1].

    Args:
        df_chunks (pd.DataFrame): Table of chunks, which should contain at least three columns that are `argument_id`, `rank`, and `polarity_score`, where `argument_id` is id of the argument a chunk belongs to, `rank` is the pagerank of a chunk within an argument, and `polarity_score` is the sentiment score of a chunk.

    Returns:
        List[float]: List of argument sentiment scores, which are floats in range [0, 1].
    """
    expected_cols = ["argument_id", "rank", "polarity_score"]
    check_columns(expected_cols=expected_cols, df=df_chunks)

    df_temp = df_chunks.groupby(by="argument_id", as_index=False).agg(
        {"rank": list, "polarity_score": list}
    )
    sentiments = df_temp.apply(lambda x: np.dot(x["rank"], x["polarity_score"]), axis=1)
    sentiments = sentiments / 2 + 0.5  # normalize to [0, 1]
    return sentiments.tolist()


def get_argument_coherence(
    scores: List[int],
    sentiments: List[float],
    min_score: int = 1,
    max_score: int = 5,
) -> List[float]:
    """Get argument coherence.

    Coherence is computed as gap between sentiments and overall scores. Overall scores are first normalized into the same range as argument sentiments, which is [0, 1]. Then their gaps are computed and applied a Gaussian kernal to transfer the value range to [0, 1].

    Args:
        scores (List[int]): List of argument overall scores.
        sentiments (List[float]): List of argument sentiment scores.
        min_score (int, optional): Lower bound of scores. Defaults to 1.
        max_score (int, optional): Upper bound of scores. Defaults to 5.

    Returns:
        List[float]: List of argument coherence scores, in range of (0, 1]
    """
    assert len(scores) == len(sentiments), "Size of scores and sentiments not match!"

    range_score = max_score - min_score
    scores = [(s - min_score) / range_score for s in scores]

    def gaussian(x):
        """Gaussian activation function."""
        return math.e ** (-(x**2) / 0.4)

    coherences = [sentiments[i] - scores[i] for i in range(len(scores))]
    coherences = list(map(gaussian, coherences))
    return coherences


def update_argument_table(
    df_arguments: pd.DataFrame,
    topics: List[List[int]],
    sentiments: List[float],
    coherences: List[float],
) -> pd.DataFrame:
    """Return a copy of argument dataframe, with new columns of argument topics, sentiments, and coherences.

    Args:
        df_arguments (pd.DataFrame): argument dataframe.
        topics (List[List[int]]): list of argument topics
        sentiments (List[float]): list of argument sentiment scores
        coherences (List[float]): list of argument coherence scores

    Returns:
        pd.DataFrame: _description_
    """
    df_copy = copy.deepcopy(df_arguments)
    df_copy["topics"] = topics
    df_copy["sentiment"] = sentiments
    df_copy["coherence"] = coherences
    return df_copy
