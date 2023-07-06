"""
Argument mining module.

This maps arguments into an attacking network based on topic modeling results.
"""

import pandas as pd
import numpy as np
from itertools import starmap, combinations
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Any, Callable, Tuple
import copy
import itertools
from numpy import nan


class ArgumentMiner:
    def __init__(self, df_arguments):
        self.df_arguments = df_arguments
    
    def select_by_topic(self, topic:int):
        """Get arguments that cover a given topic.
        """
        def check_topic_included(x):
            # value of "topics" can be string type (due to pandas operations of Orange3)
            # parse string of list into list if that is the case
            if isinstance(x, str):
                x = eval(x)
            try:
                return topic in x
            except TypeError:
                return False
        
        index = self.df_arguments["topics"].apply(check_topic_included)
        df_selection = self.df_arguments[index]
        df_selection["argument_id"] = df_selection.index
        df_selection = df_selection.reset_index(drop=True)
        return df_selection
    
    def get_edge_table(self, df_selection):
        """Given a selection of arguments, get the edge table out of it.
        """
        comb_rows = list(itertools.combinations(df_selection.index, 2))
        combs = [df_selection.loc[c, :][["score", "coherence"]].diff().iloc[1] for c in comb_rows]
        df_edges = pd.DataFrame(combs)
        df_edges[["source", "target"]] = comb_rows
        df_edges = df_edges[df_edges["score"] != 0] # only argumets with different scores 
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
        df_edges["source"] = df_selection.loc[df_edges["source"]]["argument_id"]\
            .reset_index(drop=True)
        df_edges["target"] = df_selection.loc[df_edges["target"]]["argument_id"]\
            .reset_index(drop=True)
        return df_edges
        
    def get_node_table(self, df_edges, df_nodes):
        """Given a edge table, get the node table out of it.
        """
        df_target = df_edges.groupby(by="target", as_index=False).agg({
            "source": lambda x: list(x)
        })
        
        # label arguments as attacking targets
        def label_target(row):
            sources = row["source"]
            try:
                true_count = df_target["target"].isin(sources).value_counts()[True]
            except:
                true_count = 0
            return "supportive" if true_count == len(sources) else "defeated"
        df_target["label"] = df_target.apply(label_target, axis=1)
        df_target = df_target.set_index("target")
        
        # assign labels to arguments
        def assign_label(row):
            id = row["argument_id"]
            try:
                return df_target.loc[id]["label"]
            except:
                return "supportive"
        df_nodes["label"] = df_nodes.apply(assign_label, axis=1)
        return df_nodes