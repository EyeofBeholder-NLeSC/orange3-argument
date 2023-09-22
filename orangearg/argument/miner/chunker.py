"""Argument chunker module"""

from typing import List, Tuple
import itertools
import logging

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
import networkx as nx
from textblob import TextBlob


def load_nlp_pipe(model_name: str):
    """Download the required nlp pipe if not exist

    Args:
        model_name (str): name of the nlp pipe, a full list of models can be found from https://spacy.io/usage/models.

    Returns:
        The spacy nlp model.
    """
    try:
        spacy.load(name=model_name)
    except OSError:
        spacy.cli.download(model=model_name)
    return spacy.load(name=model_name)


def get_chunk(docs: List[str]) -> Tuple[List[int], List[str]]:
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
        """Identify the sentence heads. Currently, the strategy is quite basic: it involves designating as heads all words whose dependencies are conjunctions. However, there is room for enhancement in the future."""
        heads = []
        for word in sentence.root.children:
            if word.dep_ == "conj":
                heads.append(word)
        return heads

    nlp = load_nlp_pipe(model_name="en_core_web_md")
    for i, doc in enumerate(docs):
        if doc is not None:
            doc = nlp(doc)
            for sentence in doc.sents:
                seen = set()
                heads: list[spacy.tokens.span] = find_heads(sentence)
                for head in heads:
                    head_phrase = list(head.subtree)
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
    topics = topic_model.fit_transform_reduced(chunks)
    embeds = topic_model.get_doc_embeds()
    df_topics = topic_model.get_topic_table()
    return topics, embeds, df_topics


def get_chunk_rank(arg_ids: List[int], embeds: np.ndarray):
    """In each argument, comput rank of chunks within.

    Rank can be understood as importance of chunks. This function computes the relative importance of chunks within arguments they belong to. This is done by applying the Pagerank algorithm, where similarity is computed as the cosine similarity of chunk embedding vectors.

    Args:
        arg_ids (List[int]): ids of arguments that chunks belongs to.
        embeds (np.ndarray): embedding vectors of chunks.

    Returns:
        List[float]: rank of chunks
    """

    def rank_in_argument(embeds: pd.Series):
        """Compute pagerank of embedding vectors."""
        embeds = np.array(list(embeds))
        embeds /= np.expand_dims(np.linalg.norm(embeds, axis=-1), axis=-1)
        sim_mat = embeds @ embeds.T
        graph = nx.from_numpy_array(sim_mat)
        ranks = list(nx.pagerank(graph).values())
        ranks = np.array(ranks)
        return list(ranks)

    df_embeds = pd.DataFrame({"arg_id": arg_ids, "embed": embeds.tolist()})
    df_embeds = df_embeds.groupby(by="arg_id", as_index=False).agg(
        {"embed": rank_in_argument}
    )  # group by arg_id to make the ranking within arguments
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

    Functions are implemented based on the BERTopic model. For now, the topic model is setup with a set of default parameters of the sub-models. However, it should be possible that the user can config it further. This will be a next step.

    Attributes:
        _rd_model (:obj:'UMAP'): instance of UMAP algorithm as the dimensionality reduction sub-model.
        model (:obj:'BERTopic'): the topic model that applied the sub-models predefined.
    """

    def __init__(self):
        self._rd_model = None
        self.model = None

        # initialize the topic model
        self.init_model()

    def init_model(
        self,
        transformer: str = "all-mpnet-base-v1",
        n_components: int = 5,
        min_cluster_size: int = 10,
        ngram_min: int = 1,
        ngram_max: int = 1,
    ):
        """Initialize the topic model by indicating a number of arguments.

        Args:
            transformer (str, optional): Name of the sentence embedding model. Defaults to "all-mpnet-base-v1". A list of pretrained models can be found here: https://www.sbert.net/docs/pretrained_models.html.
            n_components (int, optional): Number of dimensions after reduction. Defaults to 5.
            min_cluster_size (int, optional): Minimum size of clusters for the clustering algorithm. Defaults to 5.
            ngram_min (int, optional): Low band of ngram range for topic representation. Defaults to 1.
            ngram_max (int, optional): High band of ngram range for topic representation. Defaults to 1.
        """
        language = "english"
        nlp_pipe = "en_core_web_md"
        load_nlp_pipe(model_name=nlp_pipe)

        # initialize sub-models
        embed_model = SentenceTransformer(model_name_or_path=transformer)
        self._rd_model = UMAP(n_components=n_components)
        cluster_model = HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True)
        vector_model = CountVectorizer(
            stop_words=language, ngram_range=[ngram_min, ngram_max]
        )
        ctfidf_model = ClassTfidfTransformer(
            reduce_frequent_words=True,
        )
        represent_model = PartOfSpeech(
            model=nlp_pipe,
            pos_patterns=[[{"POS": "NOUN"}], [{"POS": "ADJ"}], [{"POS": "VERB"}]],
        )

        # initialize topic model
        self.model = BERTopic(
            language=language,
            embedding_model=embed_model,
            umap_model=self._rd_model,
            hdbscan_model=cluster_model,
            vectorizer_model=vector_model,
            ctfidf_model=ctfidf_model,
            representation_model=represent_model,
            calculate_probabilities=False,
        )

    def fit_transform_reduced(self, docs: List[str]) -> List[int]:
        """Further reduce outliers from the result of the fit_transform function.

        Note that BERTopic is a clustering approach, which means that it doesn not work if there is nothing to be clustered. And keep in mind that the input corpus should contain at least 1000 documents to get meaningful results. Refer to this thread: https://github.com/MaartenGr/BERTopic/issues/59#issuecomment-775718747.

        Args:
            docs (List[str]): The input corpus.

        Returns:
            List[int]: Topics of the input docs.
        """
        if len(docs) < 1000:
            logging.warning(
                "The input corpus should contain at least 1000 documents to get meaningful results. Got %d.",
                len(docs),
            )
        topics, _ = self.model.fit_transform(docs)
        try:
            new_topics = self.model.reduce_outliers(docs, topics, strategy="embeddings")
        except ValueError as e:
            logging.error("Failed to reduce outliers: %s", str(e))
            new_topics = topics

        self.model.update_topics(docs, topics=new_topics)
        return new_topics

    def get_topic_table(self) -> pd.DataFrame:
        """Get the table of topic information and return it as a pandas dataframe.

        Returns:
            pd.DataFrame: The topic table.
        """
        topic_info = self.model.get_topic_info()
        topic_info = topic_info.drop(["Representative_Docs"], axis=1)
        topic_info = topic_info.rename(
            columns={
                "Topic": "topic",
                "Count": "count",
                "Name": "name",
                "Representation": "keywords",
            }
        )
        topic_info["keywords"] = topic_info["keywords"].apply(tuple)
        return topic_info

    def get_doc_embeds(self) -> np.ndarray:
        """Get the embeddings of the docs.

        Returns:
            np.ndarray: Embeddings of the docs, in size of (n_doc, n_components).
        """
        return self._rd_model.embedding_
