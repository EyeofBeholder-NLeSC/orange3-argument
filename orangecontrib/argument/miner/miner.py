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
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
import torch
import networkx as nx
import spacy

from spacy.language import Language
from spacy_readability import Readability
import gensim.downloader as api
import numpy as np
from itertools import starmap, combinations
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Callable, Tuple


def sentrank(emb):
    """Compute sentence rank with the given embedding vectors.

    Args:
        emb (list): embedding vectors.
        
    Return:
        (dict): sentence index and rank score.
    """
    # compute cosine similarity matrix
    emb = torch.tensor(emb) 
    emb /= emb.norm(dim=-1).unsqueeze(-1)
    sim_mat = emb @ emb.t()
    sim_mat = sim_mat.numpy(force=True)
    
    # compute pagerank score 
    G = nx.from_numpy_array(sim_mat) 
    return nx.pagerank(G)

def chunker(docs):
    """Split argument docs into chunks by dependency parsing.

    Args:
        docs (list): input argument docs.
    """
    nlp = spacy.load('en_core_web_md')
    doc_ids = []
    chunks = []
    
    for i, doc in enumerate(docs):
        seen = set()
        doc = nlp(doc)
        for sent in doc.sents:
            heads = [c for c in sent.root.children if c.dep_ in ['conj']]
            for head in heads:
                head_chunk = [w for w in head.subtree]
                [seen.add(w) for w in head_chunk]
                chunk = ' '.join([w.text for w in head_chunk]) 
                doc_ids.append(i)
                chunks.append(chunk)
            
            unseen = [w for w in sent if w not in seen]
            chunk = ' '.join([w.text for w in unseen])
            doc_ids.append(i)
            chunks.append(chunk)
    
    return pd.DataFrame({
        'doc_id': doc_ids, 
        'chunk': chunks
    })    

class ArguTopic:
    """Build BERT-based topic modeler with given models and parameters.
    """
    EMBEDDING_MODELS = [
        'all-MiniLM-L12-v1', 
        'all-mpnet-base-v1', 
        'all-distilroberta-v1', 
        'all-roberta-large-v1', 
        'gtr-t5-large', 
        'sentence-t5-large', 
    ]
    DR_MODELS = [
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
    
    def __init__(self):
        self.model = None
    
    def set_embedding_model(self, model_name: str = 'all-mpnet-base-v1'):
        """Choose and set document embedding model.

        Args:
            model_name (str, optional): Model name. Defaults to 'all-mpnet-base-v1'.
        """
        assert model_name in self.EMBEDDING_MODELS, \
            'Embedding model not supported: %s.' % model_name
        
        self.embedding_model = SentenceTransformer(model_name)
    
    def set_dimension_reduction_model(self, model_name: str = 'UMAP', \
        n_components: int = 5):
        """Choose and set dimensionality reduction model.

        Args:
            model_name (str, optional): Model name. Defaults to 'UMAP'.
            n_components (int, optional): Number of resulting dimensions. Defaults to 5.
        """
        assert model_name in self.DR_MODELS, \
            'Dimensionality reduction model not supported: %s.' % model_name
        
        if model_name == 'UMAP': 
            self.umap_model = UMAP(n_components=n_components)
        elif model_name == 'SVD':
            self.umap_model = TruncatedSVD(n_components=n_components)
        elif model_name == 'NONE':
            self.umap_model = BaseDimensionalityReduction() 
    
    def set_clustering_model(self, model_name: str = 'HDBSCAN', \
        min_cluster_size: int = None, n_clusters: int = None):
        """Choose and set clustering model.

        Args:
            model_name (str, optional): Model name. Defaults to 'HDBSCAN'.
            min_cluster_size (int, optional): Only applicable to HDBSCAN model, miniman size of a cluster found by the algorithm. Defaults to None.
            n_clusters (int, optional): Appliable to the other models, number of clusters. Defaults to None.
        """
        assert model_name in self.CLUSTERING_MODELS, \
            'Clustering model not supported: %s.' % model_name
       
        if model_name == 'HDBSCAN':
            assert min_cluster_size is not None, \
                'Missing parameter: min_cluster_size.'
            self.hdbscan_model = HDBSCAN(
                min_cluster_size=min_cluster_size, 
                prediction_data=True
            )
        else:
            assert n_clusters is not None, \
                'Missing parameter: n_clusters.'
            if model_name == 'K-Means':
                self.hdbscan_model = KMeans(n_clusters=n_clusters)
            elif model_name == 'Agglomerative Clustering':
                self.hdbscan_model = AgglomerativeClustering(n_clusters=n_clusters)
            elif model_name == 'BIRCH':
                self.hdbscan_model = Birch(n_clusters=n_clusters)
    
    def set_topic_conduction_models(self):
        """Set topic conduction models.
        """
        class StemmedCountVectorizer(CountVectorizer):
            def build_analyzer(self):
                stemmer = SnowballStemmer('english')
                analyzer = super(StemmedCountVectorizer, self).build_analyzer()
                return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
            
        self.vectorizer_model = CountVectorizer(
            stop_words='english', 
            analyzer='word'
        )
        self.ctfidf_model = ClassTfidfTransformer(
            reduce_frequent_words=True
        )
        self.representation_model = KeyBERTInspired()
    
    def run(self, docs):
        """Perform topic modeling and return results.
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
        topics, _ = self.model.fit_transform(docs) 
        return pd.DataFrame({
            'docs': docs, 
            'topics': topics, 
        }) 
        
    def get_doc_embed(self, docs):
        """Get the document embedding vectors after dimension reduction.

        Args:
            docs (list): list of documentations

        Returns:
            numpy.array: doc embedding vectors
        """
        embed = self.model.embedding_model.embed_documents(docs)
        embed = self.model._reduce_dimensionality(embed)
        return embed

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