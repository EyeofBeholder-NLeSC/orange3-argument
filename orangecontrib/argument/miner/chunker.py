"""Argument chunker module

This splits arguments into smaller but still meaningful chunks, 
and compute topics and other scores for each chunk.
"""

from typing import List, Tuple
import copy
import itertools

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import PartOfSpeech
import spacy
import torch
import networkx as nx
from textblob import TextBlob


def get_chunk(docs: List[str]) -> Tuple[List]:
    """Split documents of a given corpus into chunks.

    A chunk can be considered as a meaningful clause, which can be part of a sentence. For instance, the sentence "I like the color of this car but it's too expensive." will be splitted as two chunks, which are "I like the color of this car" and "but it's too expensive". A dependency parser is implemented for doing this job.

    Args:
        docs (List[str]): The input corpus.

    Returns:
        List[int]: ids of the arguments that the chunks belongs to.
        List[str]: chunk text.
    """
    arg_ids, chunks = [], []

    def create_chunk(words, arg_id):
        """Merge words into chunk and add to the output lists"""
        chunk = " ".join([w.text for w in words])
        arg_ids.append(arg_id)
        chunks.append(chunk)

    def find_heads(sentence):
        """Find heads of a sentence."""
        heads = []
        for word in sentence.root.children:
            if word.dep_ == "conj":
                heads.append(word)
        return heads

    nlp = spacy.load("en_core_web_md")
    for i, doc in enumerate(docs):
        if doc is not None:
            seen = set()
            doc = nlp(doc)
            for sentence in doc.sents:
                heads = find_heads(sentence)
                for head in heads:
                    head_phrase = [w for w in head.subtree]
                    seen.update(head_phrase)
                    create_chunk(words=head_phrase, arg_id=i)
                unseen = [w for w in sentence if w not in seen]
                create_chunk(words=unseen, arg_id=i)
    return arg_ids, chunks


def get_chunk_polarity_score(chunks: List[str]):
    """Compute polarity score of each chunk in the given list.

    The polarity score is a float within the range [-1.0, 1.0], where 0 means neutral, + means positive, and - means negative.

    Args:
        chunks (List[str]): chunk list

    Returns:
        List[float]: polarity scores of the given chunks
    """
    p_scores = []
    for chunk in chunks:
        p_scores.append(TextBlob(chunk).sentiment.polarity)
    return p_scores


def get_chunk_topic(chunks: List[str]):
    """Get topic information and embedding vectors of chunks via topic modeling.

    Args:
        chunks (List[str]): chunk list.

    Returns:
        List[int]: topic ids of chunks.
        np.ndarray: embedding vectors of chunks.
        pd.DataFrame: Table of topic information.
    """
    topic_model = TopicModel()
    topic_model.init_models()
    topics = topic_model.fit_transform_reduced(chunks)
    embeds = topic_model.get_doc_embed()
    df_topics = topic_model.get_topic_table()
    return topics, embeds, df_topics


def get_chunk_rank(arg_ids: List[int], embeds: np.ndarray):
    """In each argument, comput rank of chunks within.

    Rank can be understand as importance of chunks. This way, this function compute the relative importance of chunks within arguments they belongs to. This is done by applying Pagerank algorithm, where similarity is computed as person coefficient of chunk embedding vectors.

    Args:
        arg_ids (List[int]): ids of arguments that chunks belongs to.
        embeds (np.ndarray): embedding vectors of chunks.

    Returns:
        List[float]: rank of chunks
    """

    def rank_in_argument(embeds):
        """Compute pagerank of embedding vectors."""
        embeds = list(embeds)
        embeds = torch.tensor(embeds)
        embeds /= embeds.norm(dim=-1).unsqueeze(-1)
        sim_mat = embeds @ embeds.t()
        sim_mat = sim_mat.numpy(force=True)
        graph = nx.from_numpy_array(sim_mat)
        ranks = list(nx.pagerank(graph).values())
        ranks = np.array(ranks)
        return list(ranks)

    df_embeds = pd.DataFrame({"arg_id": arg_ids, "embed": embeds.tolist()})
    df_embeds = df_embeds.groupby(by="arg_id", as_index=False).agg(
        {"embed": rank_in_argument}
    )
    ranks = df_embeds["embed"].tolist()
    ranks = list(itertools.chain(*ranks))
    return ranks


def get_chunk_table(
    arg_ids: List[int],
    chunks: List[str],
    p_scores: List[float],
    topics: List[int],
    ranks: List[float],
):
    """Given all the measures of chunks, generate and return the chunk table as a pandas dataframe, with pre-defined column names.

    Args:
        arg_ids (List[int]): ids of arguments that chunks belong to
        chunks (List[str]): chunk text
        p_scores (List[float]): polarity score of chunks
        topics (List[int]): topic id of chunks
        ranks (List[float]): rank of chunks

    Returns:
        pd.DataFrame: chunk table
    """
    return pd.DataFrame(
        {
            "argument_id": arg_ids,
            "chunk": chunks,
            "polarity_score": p_scores,
            "topic": topics,
            "rank": ranks,
        }
    )


class TopicModel:
    """Topic modeling class.

    Functions are implemented based on the BERTopic model.

    Args:
        custom_setup (dict): customized setup of the model, default to None.

    Attributes:
        setup (dict): setup of the sub-models. Defaults values::
            {
                "language": "english"
                "transformer: "all-mpnet-base-v1",
                "n_components": 5,
                "min_cluster_size": 10,
                "ngram_range": [1, 1]
            }
        embed_model (:obj:'SentenceTransformer'): instance of  the sentence transformer as the embedding sub-model.
        rd_model (:obj:'UMAP'): instance of UMAP algorithm as the dimensionality reduction sub-model.
        cluster_model (:obj:'HDBSCAN'): instance of HDBSCAN as the clustering sub-model.
        vector_model (:obj:'CountVectorizer'): instance of CountVectorizer as the vectorization sub-model.
        ctfidf_model (:obj:'ClassTfidfTransformer'): instance of ClassTfidfTransformer as the ctfidf sub-model.
        represent_model (:obj:'PartOfSpeech'): instance of PartOfSpeech as the topic representation sub-model.
        model (:obj:'BERTopic'): the topic model that applied the sub-models predefined.
    """

    def __init__(self, custom_setup: dict = None):
        """Constructor of TopicModel.

        Args:
            custom_setup (dict, optional): Customized model setup. Defaults to None.
        """
        if custom_setup is not None:
            self.setup = custom_setup
        else:
            self.setup = {
                "language": "english",
                "transformer": "all-mpnet-base-v1",
                "n_components": 5,
                "min_cluster_size": 10,
                "ngram_range": [1, 1],
            }

        # download spacy language model if not exist
        try:
            spacy.load("en_core_web_md")
        except OSError:
            spacy.cli.download(model="en_core_web_md")

        self.embed_model = None
        self.rd_model = None
        self.cluster_model = None
        self.vector_model = None
        self.ctfidf_model = None
        self.represent_model = None
        self.model = None

    def init_models(self):
        """Initialize the topic model and sub-models with the given setup."""
        # initialize sub-models
        self.embed_model = SentenceTransformer(
            model_name_or_path=self.setup["transformer"]
        )
        self.rd_model = UMAP(n_components=self.setup["n_components"])
        self.cluster_model = HDBSCAN(
            min_cluster_size=self.setup["min_cluster_size"], prediction_data=True
        )
        self.vector_model = CountVectorizer(
            stop_words="english", ngram_range=self.setup["ngram_range"]
        )
        self.ctfidf_model = ClassTfidfTransformer(
            reduce_frequent_words=True,
        )
        self.represent_model = PartOfSpeech(
            model="en_core_web_md",
            pos_patterns=[[{"POS": "NOUN"}], [{"POS": "ADJ"}], [{"POS": "VERB"}]],
        )

        # initialize topic model
        self.model = BERTopic(
            language=self.setup["language"],
            embedding_model=self.embed_model,
            umap_model=self.rd_model,
            hdbscan_model=self.cluster_model,
            vectorizer_model=self.vector_model,
            ctfidf_model=self.ctfidf_model,
            representation_model=self.represent_model,
            calculate_probabilities=False,
        )

    def fit_transform_reduced(self, docs: List[str]) -> List[int]:
        """Further reduce outliers from the result of the fit_transform function.

        Args:
            docs (List[str]): The input corpus.

        Returns:
            List[int]: Topics of the input docs.
        """
        topics, _ = self.model.fit_transform(docs)
        try:
            new_topics = self.model.reduce_outliers(
                docs, topics, strategy="distributions"
            )
        except ValueError:
            new_topics = topics

        self.model.update_topics(
            docs,
            topics=new_topics,
            vectorizer_model=self.vector_model,
            ctfidf_model=self.ctfidf_model,
            representation_model=self.represent_model,
        )
        return new_topics

    def get_topic_table(self) -> pd.DataFrame:
        """Get the table of topic information and return it as a pandas dataframe.

        Returns:
            pd.DataFrame: The topic table.
        """
        topic_info = copy.deepcopy(self.model.get_topic_info())
        topic_info = topic_info.rename(
            columns={
                "Topic": "topic",
                "Count": "count",
                "Name": "name",
                "Representation": "keywords",
                "Representative_Docs": "representative_doc",
            }
        )
        return topic_info

    def get_doc_embed(self) -> np.ndarray:
        """Get the embeddings of the docs.

        Returns:
            np.ndarray: Embeddings of the docs, in size of (n_doc, n_components).
        """
        return self.rd_model.embedding_
