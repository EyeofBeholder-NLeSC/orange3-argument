"""
Argument mining module.

Author: @jiqicn
"""

import pandas as pd
import spacy
import pytextrank
from spacy.language import Language
from spacy_readability import Readability
from importlib.util import find_spec
import gensim.downloader as api
import numpy as np
from itertools import starmap, combinations
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import networkx as nx
from typing import Tuple


@Language.component("readability")
def readability(doc):
    read = Readability()
    doc = read(doc)
    return doc


class ArgumentMiner(object):
    """
    Accept a json file of arguments and scores as input and create an attacking network.

    Attributes:
        df_arguments: a pandas DataFrame that contains the arguments and corresponding metadata.
        nlp_pipe: a spacy pip for natual language processing.
        wv_model: a word vector model of gensim.
        tokens: a numpy array that contains all tokens in the argument set.
        cluster_labels: a numpy array of token cluster labels, thus with the same size as tokens.
        network: a networkx graph instance that contains the attacking network and metadata.
    """

    df_arguments = None
    nlp_pipe = None
    wv_model = None
    tokens = None
    cluster_labels = None
    network = None

    def __init__(self, fpath: str):
        df = pd.read_json(fpath, lines=True)
        self.df_arguments = df.loc[df.astype(str).drop_duplicates().index]

    def load_nlp_pipeline(self, pipe_name: str = "en_core_web_md"):
        """
        Load the NLP pipeline that is built in spacy package.
        Will download the pipeline files if not exist.
        """
        if find_spec(pipe_name) is None:
            spacy.cli.download(pipe_name)

        self.nlp_pipe = spacy.load(pipe_name)
        self.nlp_pipe.add_pipe("textrank", last=True)
        self.nlp_pipe.add_pipe("readability", last=True)

    def load_word_vector_model(self, model_name: str = "word2vec-google-news-300"):
        """
        Load the word vector model that is built in gensim package.
        Will download the model files if not exist.
        """
        self.wv_model = api.load(model_name)

    @staticmethod
    def __get_token_ranks(
        doc: spacy.tokens.Doc, stopwords: list, trt: float = 0
    ) -> list[Tuple[str, float]]:
        """
        Get text rank of each token.
        """
        results = []
        for token in doc._.phrases:
            text = token.text
            text = text.lower().split(" ")
            text = filter(lambda x: x not in stopwords, text)
            text = " ".join(text)
            if token.rank and token.rank >= trt:
                results.append((text, token.rank))

        return results

    @staticmethod
    def __get_doc_readability(doc: spacy.tokens.Doc) -> float:
        """
        Get readability score of a document.
        """
        return doc._.flesch_kincaid_reading_ease

    def compute_ranks_and_readability(self, token_theta: float = 0):
        """
        For each argument in the input dataset, compute the token text ranks and readability.
        These data will be addd to the arguments dataframe as two columns.
        """
        if {"ranks", "readability"}.issubset(self.df_arguments.columns):
            return

        ranks = []
        readabilities = []
        stopwords = list(self.nlp_pipe.Defaults.stop_words)
        docs = self.nlp_pipe.pipe(texts=self.df_arguments["reviewText"].astype("str"))
        for doc in docs:
            ranks.append(self.__get_token_ranks(doc, stopwords, token_theta))
            readabilities.append(self.__get_doc_readability(doc))
        self.df_arguments["ranks"] = ranks
        self.df_arguments["readability"] = readabilities
        self.df_arguments = self.df_arguments[
            self.df_arguments["ranks"].astype("str") != "[]"
        ]  # remove arguments with no token
        self.df_arguments = self.df_arguments.reset_index(drop=True)

    def __compute_all_tokens(self):
        """
        Compute the full list of tokens in the arguments
        """
        if self.tokens is not None and self.tokens.size > 0:
            return

        self.tokens = []
        for doc in list(self.df_arguments["ranks"]):
            for token in doc:
                token = token[0]
                if token:
                    self.tokens.append(token)
        self.tokens = np.array(self.tokens)

    def __get_token_distances(self) -> np.ndarray:
        """
        Given a list of tokens, compute word mover's distance of all possible token pairs
        """
        assert self.tokens.size, "Should call compute_all_tokens first!"
        token_pairs = list(combinations([t.split(" ") for t in self.tokens], 2))
        token_dists = list(starmap(self.wv_model.wmdistance, token_pairs))
        token_dists = np.nan_to_num(token_dists, nan=0, posinf=100)

        return token_dists

    def __get_token_distance_matrix(self) -> pd.DataFrame:
        """
        Create token distance matrix
        """
        dist_matrix = np.zeros((len(self.tokens), len(self.tokens)))
        dist_matrix[np.triu_indices(len(self.tokens), 1)] = self.__get_token_distances()
        dist_matrix = dist_matrix + dist_matrix.T
        dist_matrix = pd.DataFrame(dist_matrix, index=self.tokens, columns=self.tokens)

        return dist_matrix

    @staticmethod
    def __cluster(dist_matrix: pd.DataFrame, k: int) -> Tuple[float, list]:
        """
        cluster the items involved in dist_matrix in k parts.
        """
        cluster = KMeans(n_clusters=k, random_state=10)
        labels = cluster.fit_predict(dist_matrix)
        try:
            silhouette = silhouette_score(dist_matrix, labels)
        except:
            silhouette = 0

        return silhouette, labels

    def __compute_cluster_labels(self):
        """
        Compute clusters of tokens
        """
        if self.cluster_labels is not None and self.cluster_labels.size:
            return

        dist_matrix = self.__get_token_distance_matrix()
        silhouette_target = -float("inf")
        for i in range(min(dist_matrix.index.size, 10)):
            silhouette, labels = self.__cluster(dist_matrix, i + 1)
            if silhouette > silhouette_target:
                silhouette_target = silhouette
                self.cluster_labels = labels

    def __get_cluster_set(self, query: np.ndarray) -> set[int]:
        """
        Find the cluster labels of all tokens in a token list and return it as a set
        """
        indices = np.isin(self.tokens, query)
        clusters = self.cluster_labels[indices]

        return set(clusters.flatten())

    def compute_clusters_and_weights(self):
        """
        Compute cluster set and weight of each argument.
        """
        if {"clusters", "weight"}.issubset(self.df_arguments.columns):
            return

        self.__compute_all_tokens()
        self.__compute_cluster_labels()

        clusters = []
        weights = []
        for i, row in self.df_arguments.iterrows():
            query = np.array([t[0] for t in row["ranks"]])
            cluster_set = self.__get_cluster_set(query)
            weight = max([t[1] for t in row["ranks"]]) * row["readability"]
            clusters.append(cluster_set)
            weights.append(weight)
        self.df_arguments.loc[self.df_arguments.index, "clusters"] = clusters
        self.df_arguments.loc[self.df_arguments.index, "weight"] = weights

    @staticmethod
    def __get_attacks(
        group_1: pd.DataFrame, group_2: pd.DataFrame
    ) -> Tuple[list, list, list]:
        """
        Given two group of arguments, get all attacks and corresponding weights
        """
        source = []
        target = []
        weight = []
        for i_1, row_1 in group_1.iterrows():
            for i_2, row_2 in group_2.iterrows():
                if (
                    row_1["clusters"] and row_2["clusters"]
                ):  # arguments having tokens from same cluster
                    if row_1["weight"] > row_2["weight"]:  # attacking direction
                        source.append(i_1)
                        target.append(i_2)
                        weight.append(row_1["weight"])
                    else:
                        source.append(i_2)
                        target.append(i_1)
                        weight.append(row_2["weight"])

        return source, target, weight

    def compute_network(self, weight_theta: int = 60):
        """
        Create the attacking network
        """
        if self.network:
            return

        df_network = {"source": [], "target": [], "weight": []}
        for curr_group in range(1, 5):
            group_1 = self.df_arguments[self.df_arguments["overall"] == curr_group]
            group_2 = self.df_arguments[self.df_arguments["overall"] > curr_group]
            if group_1.size == 0 or group_2.size == 0:
                continue
            temp_source, temp_target, temp_weight = self.__get_attacks(group_1, group_2)
            df_network["source"] += temp_source
            df_network["target"] += temp_target
            df_network["weight"] += temp_weight
        df_network = pd.DataFrame(df_network)
        df_network = df_network[df_network["weight"] >= weight_theta]
        self.network = nx.from_pandas_edgelist(
            df=df_network,
            source="source",
            target="target",
            edge_attr=["weight"],
            create_using=nx.DiGraph(),
        )

    def compute_network_node_colors(self):
        """
        Compute node colors based on its labels.
        Green means 'supported' and red means 'defeated'.

        Condition: if a node is not under attacked or all its attackers are under attacked, the node is labeled as 'supported'; otherwise, it is 'defeated'.
        """
        nodes = list(self.network.nodes)
        edges = list(self.network.edges)
        colors = {}

        targets = set([e[1] for e in edges])
        for n in nodes:
            attackers = set([e[0] for e in edges if e[1] == n])
            if not attackers or attackers & targets == attackers:
                colors[n] = "green"
            else:
                colors[n] = "red"

        nx.set_node_attributes(self.network, colors, name="color")


if __name__ == "__main__":
    fpath = "../data/sample_input.json"
    am = ArgumentMiner(fpath)
    am.load_nlp_pipeline()
    am.load_word_vector_model()
    am.compute_ranks_and_readability()
    am.compute_clusters_and_weights()
    am.compute_network()
    am.compute_network_node_colors()
