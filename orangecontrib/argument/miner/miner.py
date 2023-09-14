"""Argument mining module"""

import itertools
from ast import literal_eval
from typing import Tuple

import pandas as pd

from orangecontrib.argument.miner.utilities import check_columns


def select_by_topic(data: pd.DataFrame, topic: int) -> pd.DataFrame:
    """Select data from a dataframe so that the value in its 'topics' column contains the given topic.

    Args:
        data (pd.DataFrame): The dataframe to select data from.
        topic (int): The given topic to select.

    Returns:
        pd.DataFrame: The selection result.
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


class ArgumentMiner:
    """Argument Miner class"""

    def __init__(self, df_arguments):
        self.df_arguments = df_arguments

    def get_edge_table(self, df_selection: pd.DataFrame) -> pd.DataFrame:
        """Given a selection of arguments, get the edge table out of it."""
        comb_rows = list(itertools.combinations(df_selection.index, 2))
        combs = [
            df_selection.loc[c, :][["score", "coherence"]].diff().iloc[1]
            for c in comb_rows
        ]
        df_edges = pd.DataFrame(combs)
        df_edges[["source", "target"]] = comb_rows
        df_edges = df_edges[
            df_edges["score"] != 0
        ]  # only argumets with different scores
        df_edges = df_edges.drop("score", axis=1)
        df_edges = df_edges.rename(columns={"coherence": "weight"})

        # weight is computed as coherence_target - coherence_source by diff() above
        # so positive weight means misorder of source and target.
        def detect_direction(row):
            if row["weight"] > 0:
                temp = row["source"]
                row["source"] = row["target"]
                row["target"] = temp
            return row

        df_edges = df_edges.apply(detect_direction, axis=1)
        df_edges["weight"] = df_edges["weight"].abs()
        df_edges = df_edges.reset_index(drop=True)

        # refill source and target with argument id instead of df index
        df_edges["source"] = df_selection.loc[df_edges["source"]][
            "argument_id"
        ].reset_index(drop=True)
        df_edges["target"] = df_selection.loc[df_edges["target"]][
            "argument_id"
        ].reset_index(drop=True)

        return df_edges

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
