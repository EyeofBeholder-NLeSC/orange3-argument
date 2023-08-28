"""Argument chunker module

This splits arguments into smaller but still meaningful chunks, 
and compute topics and other scores for each chunk.
"""

from typing import List
import copy
import itertools

import pandas as pd
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
import numpy as np
from textblob import TextBlob


class ArgumentChunker:
    """Split arguments into smaller but meaningful chunks."""

    def __init__(self, docs: List[str]):
        self.docs = docs
        self.df_chunks = None

        # download the required model if not existed
        try:
            spacy.load("en_core_web_md")
        except OSError:
            spacy.cli.download("en_core_web_md")

        self.topic_model = ArgumentTopic()

    def chunk(self):
        """Split argument into chunks. Dependency parsing."""
        if self.df_chunks is not None and "chunk" in self.df_chunks.columns:
            return

        nlp = spacy.load("en_core_web_md")
        argt_ids = []
        chunks = []
        for i, doc in enumerate(self.docs):
            if doc is None:
                continue
            seen = set()
            doc = nlp(doc)
            for sent in doc.sents:
                heads = [c for c in sent.root.children if c.dep_ in ["conj"]]
                for head in heads:
                    head_chunk = [w for w in head.subtree]
                    for word in head_chunk:
                        seen.add(word)
                    chunk = " ".join([w.text for w in head_chunk])
                    argt_ids.append(i)
                    chunks.append(chunk)
                unseen = [w for w in sent if w not in seen]
                chunk = " ".join([w.text for w in unseen])
                argt_ids.append(i)
                chunks.append(chunk)

        self.df_chunks = pd.DataFrame({"argument_id": argt_ids, "chunk": chunks})

    def chunk_polarity_score(self):
        """Compute chunk polarity scores."""
        if "polarity_score" in self.df_chunks.columns:
            return

        self.df_chunks["polarity_score"] = self.df_chunks["chunk"].apply(
            lambda c: TextBlob(c).sentiment.polarity
        )

    def chunk_topic(self):
        """Compute chunk topics and probabilities."""
        if "topic" in self.df_chunks.columns:
            return

        topics, _ = self.topic_model.fit_transform_reduced(self.df_chunks["chunk"])
        self.df_chunks["topic"] = topics

    def chunk_rank(self):
        """Compute sentence rank of chunks."""
        if "rank" in self.df_chunks.columns:
            return
        assert (
            "topic" in self.df_chunks.columns
        ), "Should do topic modeling before computing chunk ranks!"

        def rank_in_argument(embeds):
            embeds = list(embeds)
            embeds = torch.tensor(embeds)
            embeds /= embeds.norm(dim=-1).unsqueeze(-1)
            sim_mat = embeds @ embeds.t()
            sim_mat = sim_mat.numpy(force=True)
            graph = nx.from_numpy_array(sim_mat)
            ranks = list(nx.pagerank(graph).values())
            ranks = np.array(ranks)
            return list(ranks)

        embeds = self.topic_model.get_doc_embed()
        df_temp = pd.DataFrame(
            {"argument_id": self.df_chunks["argument_id"], "embed": embeds.tolist()}
        )
        df_temp = df_temp.groupby(by="argument_id", as_index=False).agg(
            {"embed": rank_in_argument}
        )
        ranks = df_temp["embed"].tolist()
        ranks = list(itertools.chain(*ranks))
        self.df_chunks["rank"] = ranks

    def get_chunk_table(self):
        """Get full info table of chunks.

        This table will include the following columns: chunk, argument_id,
        topic, topic_prob, rank, polarity_score.
        """
        self.chunk()
        self.chunk_topic()
        self.chunk_rank()
        self.chunk_polarity_score()
        return copy.deepcopy(self.df_chunks)

    def get_topic_table(self):
        """Get topic info table.

        This table will include the following columns: topic, name, count,
        keywords, keyword_scores.
        """
        assert (
            "topic" in self.df_chunks.columns
        ), "Should do topic modeling before getting topic info table!"
        return self.topic_model.get_topic_table()


class ArgumentTopic(BERTopic):
    """
    BERT-based topic modeling and doc ranking.
    """

    def __init__(self):
        self.embedding_model = SentenceTransformer("all-mpnet-base-v1")
        self.umap_model = UMAP(n_components=5)
        self.hdbscan_model = HDBSCAN(min_cluster_size=10, prediction_data=True)
        self.vectorizer_model = CountVectorizer(
            stop_words="english", ngram_range=[1, 1]
        )
        self.ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        self.representation_model = PartOfSpeech(
            "en_core_web_md",
            pos_patterns=[[{"POS": "NOUN"}], [{"POS": "ADJ"}], [{"POS": "VERB"}]],
        )

        super().__init__(
            embedding_model=self.embedding_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            vectorizer_model=self.vectorizer_model,
            ctfidf_model=self.ctfidf_model,
            representation_model=self.representation_model,
            calculate_probabilities=False,
        )

    def fit_transform_reduced(self, docs: List[str]):
        """Fit documents and reduce outliers by default."""
        topics, probs = self.fit_transform(docs)
        try:
            new_topics = self.reduce_outliers(docs, topics, strategy="distributions")
        except ValueError:
            new_topics = topics

        self.update_topics(
            docs,
            topics=new_topics,
            vectorizer_model=self.vectorizer_model,
            ctfidf_model=self.ctfidf_model,
            representation_model=self.representation_model,
        )

        return new_topics, probs

    def get_topic_table(self):
        """Return topic information as a table."""
        topic_info = copy.deepcopy(self.get_topic_info())
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

    def get_doc_embed(self):
        """Get document embeddings after dimensionality reduction."""
        return self.umap_model.embedding_
