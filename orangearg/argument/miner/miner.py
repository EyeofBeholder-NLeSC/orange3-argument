"""Argument mining module"""
from ast import literal_eval
from typing import Tuple, List
from itertools import combinations

import pandas as pd

from orangearg.argument.miner.utilities import check_columns


def select_by_topic(data: pd.DataFrame, topic: int) -> pd.DataFrame:
    """Select arguments mentioning the given topic.

    Args:
        data (pd.DataFrame): The argument dataframe that must contain the 'topics' column.
        topic (int): The given topic to select.

    Raises:
        ValueError: if the 'topics' value of an argument is stored as something else other than a tuple (e.g. a list).

    Returns:
        pd.DataFrame: Part of the original argument dataframe that only contains arguments mentioning the given topic.
    """
    expected_cols = ["topics"]
    check_columns(expected_cols=expected_cols, data=data)

    select_condition = []
    for i, row in data.iterrows():
        topics = literal_eval(str(row["topics"]))
        if isinstance(topics, int):  # in case of tuple of one item
            select_condition.append(topic == topics)
        elif isinstance(topics, tuple):
            select_condition.append(topic in topics)
        else:
            raise ValueError(
                f"Topics of the {i}th argument should be a tuple, but {topics} is given."
            )

    selection = data[select_condition]
    selection["argument_id"] = selection.index
    return selection.reset_index(drop=True)


def get_edges(data: pd.DataFrame) -> List[Tuple[int]]:
    """Get edges from argument dataframe.

    Edges (attacks) only exist if the two arguments have different overall scores. Edges are tuple of source and target, which are indices of the corresponding argument in the input dataframe.

    Args:
        data (pd.DataFrame): The argument dataframe that must have the 'score' column.

    Returns:
        List[Tuple[int]]: The edge list.
    """
    expected_cols = ["score"]
    check_columns(expected_cols=expected_cols, data=data)

    id_combs = list(combinations(data.index, 2))
    edges = []
    for id_combo in id_combs:
        if data.loc[id_combo[0]]["score"] != data.loc[id_combo[1]]["score"]:
            edges.append(id_combo)
    return edges


def get_edge_weights(data: pd.DataFrame, edges: List[Tuple[int]]) -> List[float]:
    """Get edge weights.

    Edge weights are computed as the difference between the coherence of the source and that of the target.

    Args:
        data (pd.DataFrame): The argument dataframe that must have the 'coherence' column.
        edges (List[Tuple[int]]): The edge list.

    Returns:
        List[float]: The list of edge weights.
    """
    expected_cols = ["coherence"]
    check_columns(expected_cols=expected_cols, data=data)

    weights = []
    for s, t in edges:
        weight = data.loc[s]["coherence"] - data.loc[t]["coherence"]
        weights.append(round(weight, 2))
    return weights


def get_edge_table(edges: List[Tuple[int]], weights: List[float]) -> pd.DataFrame:
    """Get the edge dataframe.

    There will be three columns in the output dataframe, which are 'source', 'target', and 'weight'. Together, they describe weighted directed edges from source to target argument. Note that there will be no negative weights in the output dataframe, instead, all values will be replace with their absolute values. For edges with negative weights, we swap their source and target.

    Args:
        edges (List[Tuple[int]]): The edge list, which are tuples of source and target argument ids.
        weights (List[float]): The list of edge weights.

    Raises:
        ValueError: if size of the input lists doesn't match.

    Returns:
        pd.DataFrame: The result edge dataframe.
    """
    if len(edges) != len(weights):
        raise ValueError(
            f"Length of edges and weigts are not equal: {(len(edges), len(weights))}."
        )

    for i, w in enumerate(weights):
        if w < 0:
            edges[i] = (edges[i][1], edges[i][0])
    result = pd.DataFrame(edges, columns=["source", "target"])
    result["weight"] = [abs(w) for w in weights]
    return result


def get_node_labels(
    indices: List[int], sources: List[int], targets: List[int]
) -> List[str]:
    """Get labels of arguments given the attacking network.

    Arguments are separated into two classes, 'supportive' and 'defeated', which generally means reliable and unreliable. The rule of detecting the labels is as follows: if an argument is attacked by another argument who is not attacked by any argument, then this argument is labeled as 'defeated'; otherwise, it's labeled as 'supportive'. That means, if an argument appears in `targets`, where its corresponding source doesn't, this argument will be labeled as 'defeated', and otherwise 'supportive'.

    Args:
        indices (List[int]): The node index list
        sources (List[int]): The source list of the attacking network.
        targets (List[int]): The target list of the attacking network.

    Returns:
        List[str]: The label list.
    """
    labels = {i: "supportive" for i in indices}

    for i, target in enumerate(targets):
        source = sources[i]
        if source not in targets:
            labels[target] = "defeated"

    return list(labels.values())


def get_node_table(
    arg_ids: List[int], arguments: List[str], scores: List[int], labels: List[str]
) -> pd.DataFrame:
    """Get the node dataframe.

    The node dataframe will contain 4 columns, that are 'argument_id', 'argument', 'score', and 'label'.

    Args:
        arg_ids (List[int]): The argument id list.
        arguments (List[str]): The argument text list.
        scores (List[int]): The list of argument overall score.
        labels (List[str]): The argument label list.

    Returns:
        pd.DataFrame: The result node dataframe.
    """
    return pd.DataFrame(
        {
            "argument_id": arg_ids,
            "argument": arguments,
            "score": scores,
            "label": labels,
        }
    )
