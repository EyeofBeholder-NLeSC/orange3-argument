"""
Argument mining module.

Author: @jiqicn
"""

import pandas as pd
import spacy
import pytextrank
from spacy.language import Language
from spacy_readability import Readability
from flair.nn import Classifier
from flair.data import Sentence
from importlib.util import find_spec
import gensim.downloader as api
import numpy as np
from itertools import starmap, combinations
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Tuple


@Language.component("readability")
def readability(doc):
    read = Readability()
    doc = read(doc)
    return doc


class ArgumentProcessor(object):
    """Process argument before mining.
    
    - Rename argument and score columns; 
    - Label arguments by usefulness.

    Args:
        object (_type_): _description_
    """
  
    def __init__(self, df, lang: str = "en"):
        if df is None:
            raise TypeError("Input data frame must not be None!")
        
        lang2name = {"en": "en_core_web_md"}
        pipe_name = lang2name[lang]
        if find_spec(pipe_name) is None:
            spacy.cli.download(pipe_name)
            
        self.df = df
        self.nlp_pipe = spacy.load(pipe_name)
        self.nlp_pipe.add_pipe('textrank', last=True)
        self.nlp_pipe.add_pipe('readability', last=True)
        self.sa_pipe = Classifier.load('sentiment')
            
    def rename_column(self, old_name, new_name):
        """Rename a column with old_name to new_name.
        """ 
        self.df.rename({old_name: new_name}, axis=1, inplace=True)
        
    def compute_textrank(self, theta: float = 0):
        """Compute textrank of tokens in each argument.
        """
        if 'ranks' in self.df.columns:
            return
        
        stopwords = self.nlp_pipe.Defaults.stop_words
        stopwords = list(stopwords) 
        docs = self.df['argument'].astype('str')
        docs = self.nlp_pipe.pipe(texts=docs)
        
        def get_token_rank(doc): 
            """Get list of (text, textrank) of all tokens in a document
            """
            result = []
            for token in doc._.phrases:
                rank = token.rank
                if rank and rank >= theta:
                    text = token.text 
                    text = text.lower().split(' ')
                    text = filter(lambda x: x not in stopwords, text)
                    text = ' '.join(text)
                    result.append((text, rank))
            return result
       
        ranks = []
        for doc in docs:
            ranks.append(get_token_rank(doc))
        self.df['ranks'] = ranks
        self.df['ranks'] = self.df['ranks'].astype('str')
    
    def compute_readability(self, readable_theta: int = 4):
        """Compute readability of each argument.
        """
        if 'readability' in self.df.columns:
            return
       
        docs = self.df['argument'].astype('str')
        docs = self.nlp_pipe.pipe(texts=docs)
        
        readabilities = [] 
        for doc in docs:
            readability = doc._.flesch_kincaid_reading_ease
            readabilities.append(readability)
        v_min = min(readabilities)
        v_max = max(readabilities)    
        readabilities = [(r - v_min) * 10 / (v_max - v_min) for r in readabilities] 
        self.df['readability'] = readabilities
        self.df['readable'] = [r >= readable_theta for r in readabilities]
        
    def compute_sentiment(self):
        """Compute sentiment of each argument.
        """
        if 'sentiment' in self.df.columns:
            return 
        
        sentiments = [] 
        for index, row in self.df.iterrows():
            text = row['argument'] 
            sentence = Sentence(text)
            self.sa_pipe.predict(sentence)
            sentence = str(sentence)
            if 'POSITIVE' in sentence:
                sentiments.append(1)
            elif 'NEGATIVE' in sentence:
                sentiments.append(-1)
            else:
                sentiments.append(0)
        self.df['sentiment'] = sentiments
    
    def compute_coherence(self):
        """Compute coherence between the argument sentiment and its score
        """
        if 'coherence' in self.df.columns:
            return
        
        # this is based on the way how the flair sentiment model works    
        score2sentiment = {
            1: -1, 
            2: -1,
            3: -1, 
            4: 1, 
            5: 1
        }
        
        coherence = []
        for index, row in self.df.iterrows():
            c = row['sentiment'] == score2sentiment[row['score']]
            coherence.append(c)
        self.df['coherence'] = coherence
    
    def compute_usefulness(self):
        """Compute usefulness (0~2) of each argument. 
        """
        if 'usefulness' in self.df.columns:
            return
        
        usefulness = []  
        for index, row in self.df.iterrows():
            u = row['readable'] + row['coherence']
            usefulness.append(u)
        self.df['usefulness'] = usefulness
        
    def filter(self, usefulness_theta: int = 2):
        """Filter out the arguments that are not of interest.
        """
        assert 0 <= usefulness_theta <= 2, "Usefulness should be in range [0, 2]!"
       
        result = self.df.copy()
        result = result[result['ranks'].astype(bool)] 
        result = result[result['usefulness'] >= usefulness_theta]
        result = result.reset_index(drop=True)
        return result


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
        
        self.df = df
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