"""Argument mining module"""
from ast import literal_eval
from typing import Tuple, List
from itertools import combinations

import pandas as pd

from orangecontrib.argument.miner.utilities import check_columns


def select_by_topic(data: pd.DataFrame, topic: int) -> pd.DataFrame:
    """Select arguments mentioning the given topic.

    Args:
        data (pd.DataFrame): The argument dataframe that must contain the 'topic' column.
        topic (int): The given topic to select.

    Returns:
        pd.DataFrame: Part of the original argument dataframe that only contains arguments mentioning the given topic.
    """
    expected_cols = ["topics"]
    check_columns(expected_cols=expected_cols, data=data)

    def check_topic_included(topics: Tuple[int]) -> bool:
        try:
            return topic in topics
        except TypeError:
            topics = literal_eval(str(topics))
            if isinstance(topics, int):
                return topic == topics
            elif isinstance(topics, tuple):
                return topic in topics

    selection_indices = data["topics"].apply(check_topic_included)
    return data[selection_indices]


def get_edges(data: pd.DataFrame) -> List[Tuple[int]]:
    """Get edges from argument dataframe.

    Edges (attackness) only exist if the two arguments have different overall scores.

    Args:
        data (pd.DataFrame): The argument dataframe that must have the 'score' column.

    Returns:
        List[Tuple[int]]: The edge list, which are tuples of source and target argument ids.
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
        edges (List[Tuple[int]]): The edge list, which are tuples of source and target argument ids.

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

    There will be three columns in the output dataframe, which are 'source', 'target', and 'weight'. Together, they describe weighted directed edges from source to target argument. Note that there will be no negative weights in the output dataframe, instead, all values will be replace with their absolution values. For edges with negative weights, we swap their source and target.

    Args:
        edges (List[Tuple[int]]): The edge list, which are tuples of source and target argument ids.
        weights (List[float]): The list of edge weights.

    Returns:
        pd.DataFrame: The result edge dataframe.
    """
    for i, w in enumerate(weights):
        if w < 0:
            edges[i] = (edges[i][1], edges[i][0])
    result = pd.DataFrame(edges, columns=["source", "target"])
    result["weight"] = [abs(w) for w in weights]
    return result


def get_node_labels():
    pass


def get_node_table():
    pass


class ArgumentMiner:
    """Argument Miner class"""

    def __init__(self, df_arguments):
        self.df_arguments = df_arguments

    def get_node_table(
        self, df_edges: pd.DataFrame, df_nodes: pd.DataFrame
    ) -> pd.DataFrame:
        """Given a edge table, get the node table out of it."""
        df_target = df_edges.groupby(by="target", as_index=False).agg({"source": list})

        # label arguments as attacking targets
        def label_target(row):
            sources = row["source"]
            try:
                true_count = df_target["target"].isin(sources).value_counts()[True]
            except KeyError:
                true_count = 0
            return "supportive" if true_count == len(sources) else "defeated"

        df_target["label"] = df_target.apply(label_target, axis=1)
        df_target = df_target.set_index("target")

        # assign labels to arguments
        def assign_label(row):
            arg_id = row["argument_id"]
            try:
                return df_target.loc[arg_id]["label"]
            except KeyError:
                return "supportive"

        df_nodes["label"] = df_nodes.apply(assign_label, axis=1)
        return df_nodes

    def map_edge_tables(
        self, df_edges: pd.DataFrame, df_nodes: pd.DataFrame
    ) -> pd.DataFrame:
        """Map source and target in df_edges to indices of nodes in df_nodes."""
        mapper = df_nodes["argument_id"]
        mapper = pd.Series(mapper.index.values, mapper)
        df_edges["source"] = df_edges["source"].apply(lambda x: mapper[x])
        df_edges["target"] = df_edges["target"].apply(lambda x: mapper[x])

        return df_edges
