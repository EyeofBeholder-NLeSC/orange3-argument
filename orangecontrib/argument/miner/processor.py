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


def argument_coherence(self):
    """Compute argument coherence."""
    if "coherence" in self.df_arguments.columns:
        return
    assert "sentiment" in self.df_arguments.columns, "Should compute sentiment first!"

    def gaussian(x):
        """Gaussian activation function."""
        return math.e ** (-(x**2) / 0.4)

    max_score = self.df_arguments["score"].max() - 1
    coherences = (
        self.df_arguments["sentiment"] - (self.df_arguments["score"] - 1) / max_score
    ).apply(gaussian)
    self.df_arguments["coherence"] = coherences


def get_argument_table(self, df_chunks: pd.DataFrame) -> pd.DataFrame:
    """Get the processed argument table."""
    self.argument_topics(df_chunks)
    self.argument_sentiment(df_chunks)
    self.argument_coherence()
    return copy.deepcopy(self.df_arguments)
