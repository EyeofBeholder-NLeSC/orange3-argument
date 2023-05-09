"""
Argument mining module.

Author: @jiqicn
"""

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.decomposition import TruncatedSVD
from bertopic.dimensionality import BaseDimensionalityReduction
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch

from spacy.language import Language
from spacy_readability import Readability
import gensim.downloader as api
import numpy as np
from itertools import starmap, combinations
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Tuple


class ArgumentMinerBERT:
    """Argument mining module based on BERT-based topic modeling.
    """
    EMBEDDING_MODELS = [
        'all-MiniLM-L12-v1', 
        'all-mpnet-base-v1', 
        'all-distilroberta-v1', 
        'all-roberta-large-v1', 
        'distiluse-base-multilingual-cased-v1', 
        'paraphrase-multilingual-MiniLM-L12-v2', 
        'paraphrase-multilingual-mpnet-base-v2', 
        'gtr-t5-large', 
        'sentence-t5-large', 
        'average_word_embeddings_glove.6B.300d',
        'average_word_embeddings_komninos'
    ]
    DIMENSION_REDUCTION_MODELS = [
        'UMAP', 
        'SVD', 
        'NONE'
    ]
    CLUSTERING_MODELS = [
        'HDBSCAN', 
        'K-Means', 
        'Agglomerative Clustering', 
        'BIRCH'
    ]
    
    def __init__(self, df: pd.DataFrame):
        self.model = None
        self.df = df
       
    def set_embedding_model(self, model_name: str = 'all-mpnet-base-v1'):
        """Choose and setup the embedding model.
        """
        assert model_name in self.EMBEDDING_MODELS, 'Embedding model not found: %s' % model_name
        
        self.embedding_model = SentenceTransformer(model_name)
    
    def set_dimension_reduction_model(self, model_name: str = 'UMAP', n_components: int = 5):
        """Choose and setup the dimensionality reduction model
        """
        assert model_name in self.DIMENSION_REDUCTION_MODELS, 'Dimensionality reduction model not found: %s' % model_name
        
        if model_name == 'UMAP': 
            self.umap_model = UMAP(n_components=n_components)
        elif model_name == 'SVD':
            self.umap_model = TruncatedSVD(n_components=n_components)
        elif model_name == 'NONE':
            self.umap_model = BaseDimensionalityReduction() 
    
    def set_clustering_model(self, model_name: str = 'HDBSCAN', min_cluster_size: int = None, n_clusters: int = None):
        """Choose and setup the clustering model.
        """
        assert model_name in self.CLUSTERING_MODELS, 'Clustering model not found: %s' % model_name
       
        if model_name == 'HDBSCAN':
            assert min_cluster_size is not None, 'Fail to set model parameter: %s' % model_name
            self.hdbscan_model = HDBSCAN(
                min_cluster_size=min_cluster_size, 
                prediction_data=True
            )
        else:
            assert n_clusters is not None, 'Fail to set model parameter: %s' % model_name 
            if model_name == 'K-Means':
                self.hdbscan_model = KMeans(n_clusters=n_clusters)
            elif model_name == 'Agglomerative Clustering':
                self.hdbscan_model = AgglomerativeClustering(n_clusters=n_clusters)
            elif model_name == 'BIRCH':
                self.hdbscan_model = Birch(n_clusters=n_clusters)
    
    def set_topic_conduction_models(self, **kwargs):
        """Set parameters for tokenizer, c-tf-if, and representation models.
        """
        self.vectorizer_model = None
        self.ctfidf_model = None
        self.representation_model = None
    
    def do_topic_modeling(self):
        """Perform topic modeling with a given setup.
        """
        self.model = BERTopic(
            embedding_model=self.embedding_model, 
            umap_model=self.umap_model, 
            hdbscan_model=self.hdbscan_model, 
            vectorizer_model=self.vectorizer_model, 
            ctfidf_model=self.ctfidf_model, 
            representation_model=self.representation_model, 
            calculate_probabilities=True # compute probabilities of all topics cross all docs
        )
        
        topic, probs = self.model.fit_transform(self.df['argument']) 
        self.df['topics'] = topic 
        self.df['topic_distr'] = probs
    
    def get_topics(self):
        """Get the topic information table.
        """
        pass

class ArgumentMiner(object):
    """
    Accept a json file of arguments and scores as input and create an attacking network.

    Attributes:
        df_arguments: a pandas DataFrame that contains the arguments and corresponding metadata.
        nlp_pipe: a spacy pip for natual language processing.
        wv_model: a word vector model of gensim.
        tokens: a numpy array that contains all tokens in the argument set.
        cluster_labels: a numpy array of token cluster labels, thus with the same size as tokens.
        df_edge: edge table of the attacking network.
        df_node: node table of the attacking network.
    """

    df_arguments = None
    wv_model = None
    tokens = None
    cluster_labels = None
    df_edge = None
    df_node = None

    def __init__(self, df: pd.DataFrame = None, lang: str ='en'):
        lang2name = {'en': 'word2vec-google-news-300'}
        model_name = lang2name[lang]
        
        self.df_arguments = df
        self.wv_model = api.load(model_name)
            
    def __compute_all_tokens(self):
        """
        Compute the full list of tokens in the arguments
        """
        if self.tokens is not None and self.tokens.size > 0:
            return

        self.tokens = []
        for doc in list(self.df_arguments["ranks"]):
            doc = eval(doc) # parse str to list
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
            ranks = eval(row['ranks'])
            query = np.array([t[0] for t in ranks])
            cluster_set = self.__get_cluster_set(query)
            weight = max([t[1] for t in ranks]) * row["readability"]
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

    # TODO: remove instance variable df_edge and df_node, but just return values
    def compute_edge_table(self):
        """
        Compute the edge table of the attacking network.
        """
        if self.df_edge:
            return
        else: 
            self.df_edge = {"source": [], "target": [], "weight": []}
        
        for curr_group in range(1, 5):
            group_1 = self.df_arguments[self.df_arguments["score"] == curr_group]
            group_2 = self.df_arguments[self.df_arguments["score"] > curr_group]
            if group_1.size == 0 or group_2.size == 0:
                continue
            temp_source, temp_target, temp_weight = self.__get_attacks(group_1, group_2)
            self.df_edge["source"] += temp_source
            self.df_edge["target"] += temp_target
            self.df_edge["weight"] += temp_weight
        self.df_edge = pd.DataFrame(self.df_edge)
        
        w_min, w_max = self.df_edge['weight'].min(), self.df_edge['weight'].max()
        if w_min == w_max:
            self.df_edge.assign(weight=1.0)
        else:
            self.df_edge['weight'] = (self.df_edge['weight'] - w_min) / (w_max - w_min)
    
    def compute_node_table(self):
        """Compute node table of the attacking network.
        
        label can be 'supportive' or 'defeated'.
        """
        if self.df_node:
            return
        else:
            self.df_node = {'argument': [], 'score': [], 'label': []}
        
        targets = set(self.df_edge['target'].tolist())
        for i, row in self.df_arguments.iterrows():
            attackers = self.df_edge[self.df_edge['target'] == i]['source']
            attackers = set(attackers.tolist())
            self.df_node['argument'].append(row['argument']) 
            self.df_node['score'].append(row['score'])
             
            if not attackers or attackers & targets == attackers:
                self.df_node['label'].append('supportive')
            else:
                self.df_node['label'].append('defeated')
                
        self.df_node = pd.DataFrame(self.df_node)