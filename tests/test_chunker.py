import pandas as pd
import numpy as np
import pytest
from chunker import ArgumentTopic, ArgumentChunker


@pytest.fixture(scope="session")
def chunker(input_df):
    yield ArgumentChunker(input_df["argument"])

@pytest.fixture(scope="session")
def topic_model():
    yield ArgumentTopic() 


class TestArgumentTopic:
    def test_fit_transform_reduced(self, topic_model, input_df):
        docs = input_df["argument"]
        topics, probs = topic_model.fit_transform_reduced(docs)

        assert isinstance(topics, list)
        assert isinstance(probs, np.ndarray)
        assert len(topics) == len(input_df)

    def test_get_topic_table(self, topic_model):
        topic_table = topic_model.get_topic_table()

        assert isinstance(topic_table, pd.DataFrame)
        assert set(topic_table.columns) == {"topic", "name", "count", "keywords", "keyword_scores"}
        assert len(topic_table) > 0

    def test_get_doc_embed(self, topic_model, input_df):
        doc_embed = topic_model.get_doc_embed()

        assert isinstance(doc_embed, np.ndarray)
        assert len(doc_embed) == len(input_df)
        assert doc_embed.shape[1] == 5  # Embedding dimension   


class TestArgumentChunker:    
    def test_chunk(self, chunker):
        chunker.chunk()
        df_chunks = chunker.df_chunks

        assert isinstance(df_chunks, pd.DataFrame)
        assert set(df_chunks.columns) == {"argument_id", "chunk"}

    def test_chunk_polarity_score(self, chunker):
        chunker.chunk_polarity_score()
        df_chunks = chunker.df_chunks

        assert "polarity_score" in df_chunks.columns

    def test_chunk_topic(self, chunker):
        chunker.chunk_topic()
        df_chunks = chunker.df_chunks

        assert "topic" in df_chunks.columns

    def test_chunk_rank(self, chunker):
        chunker.chunk_rank()
        df_chunks = chunker.df_chunks

        assert "rank" in df_chunks.columns

    def test_get_chunk_table(self, chunker):
        df_chunks = chunker.get_chunk_table()

        assert isinstance(df_chunks, pd.DataFrame)
        assert set(df_chunks.columns) == {"argument_id", "chunk", "topic", "rank", "polarity_score"}
        assert len(df_chunks) > 0

    def test_get_topic_table(self, chunker):
        df_topic = chunker.get_topic_table()

        assert isinstance(df_topic, pd.DataFrame)
        assert set(df_topic.columns) == {"topic", "name", "count", "keywords", "keyword_scores"}
        assert len(df_topic) > 0