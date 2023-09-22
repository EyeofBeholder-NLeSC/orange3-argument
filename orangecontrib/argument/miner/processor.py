"""Argument processor module."""

from typing import List, Dict, Tuple
import copy
import math
from collections import defaultdict

import numpy as np
import pandas as pd


def _match_list_size(*args: List):
    """With an arbitrary number of lists as input, check if they are in the same size."""
    if any(len(arg) != len(args[0]) for arg in args):
        raise ValueError(f"Input size not match: {args}.")


def _aggregate_list_by_another(keys: List, values: List) -> Dict:
    """Aggregate a list according to elements of another list.

    Args:
        keys (List): The group keys.
        values (List): The list to be aggregated.

    Returns:
        Dict: The aggregation result.
    """
    result = defaultdict(list)
    for i, key in enumerate(keys):
        result[key].append(values[i])
    return result


def get_argument_topics(arg_ids: List[int], topics: List[int]) -> List[Tuple[int]]:
    """Get argument topics.

    The topics of an argument is a combination of the topics of all chunks that belong to this argument. Duplications are not removed, and the reason behind is that duplications can be treated as a sign of topic importance. Also, even though two chunks can belong to the same topic, they could still have different ranks within an argument.

    Args:
        arg_ids (List[int]): the argument ids of chunks.
        topics (List[int]): the topic indices of chunks.

    Returns:
        List[list[int]]: list of argument topics, which is also a list containing topic indices of chunks belonging to this argument.
    """
    _match_list_size(arg_ids, topics)
    result = _aggregate_list_by_another(keys=arg_ids, values=topics)
    result = result.values()
    return [tuple(r) for r in result]


def get_argument_sentiment(
    arg_ids: List[int],
    ranks: List[float],
    p_scores: List[float],
    min_sent: int = -1,
    max_sent: int = 1,
) -> List[float]:
    """Get argument sentiment score.

    The sentiment score of an argument is calculated as a weighted sum of sentiment scores of chunks belonging to this argument, where weights are ranks of the chunks. The result score is then normalized into range [0, 1].

    Args:
        arg_ids (List[int]): the argument ids of chunks.
        ranks (List[float]): the pagerank of chunks within arguments.
        p_scores (List[float]): the sentiment polarity scores of chunks.
        min_sent (int): minimun of argument sentiment before normalization. Defaults to -1.
        max_sent (int): maximum of argument sentiment before normalization. Defaults to 1.

    Returns:
        List[float]: List of argument sentiment scores, which are floats in range [0, 1].
    """
    _match_list_size(arg_ids, ranks, p_scores)

    grouped_ranks = _aggregate_list_by_another(keys=arg_ids, values=ranks)
    grouped_p_scores = _aggregate_list_by_another(keys=arg_ids, values=p_scores)

    sentiments = []
    for arg_id, rank in grouped_ranks.items():
        p_score = grouped_p_scores[arg_id]
        sentiment = np.dot(rank, p_score)
        sentiment = (sentiment - min_sent) / (max_sent - min_sent)
        sentiments.append(sentiment)
    return sentiments


def get_argument_coherence(
    scores: List[int],
    sentiments: List[float],
    min_score: int = 1,
    max_score: int = 5,
    variance: float = 0.2,
) -> List[float]:
    """Get argument coherence.

    Coherence is computed as inversed difference between sentiments and overall scores. Overall scores are first normalized into the same range as argument sentiments, which is [0, 1]. Then their differences are computed and applied a Gaussian kernal to invert and scale the differences to [0, 1].

    Args:
        scores (List[int]): List of argument overall scores.
        sentiments (List[float]): List of argument sentiment scores.
        min_score (int, optional): Lower bound of scores. Defaults to 1.
        max_score (int, optional): Upper bound of scores. Defaults to 5.
        variance (float): variance of the Gaussian kernal.

    Returns:
        List[float]: List of argument coherence scores, in range of (0, 1]
    """
    _match_list_size(sentiments, scores)

    range_score = max_score - min_score
    scores = [(s - min_score) / range_score for s in scores]

    def gaussian(x):
        """Gaussian activation function."""
        return math.e ** (-(x**2) / (2 * variance))

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
