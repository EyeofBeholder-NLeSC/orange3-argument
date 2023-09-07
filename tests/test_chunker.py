"""Tests of the chunker module"""
from difflib import SequenceMatcher
import csv
from unittest.mock import Mock

import pytest
import numpy as np

from orangecontrib.argument.miner.chunker import (
    get_chunk,
    get_chunk_polarity_score,
    get_chunk_rank,
    get_chunk_topic,
    get_chunk_table,
    TopicModel,
)


@pytest.fixture(scope="function")
def large_chunk_set():
    """Chunk set for integration test"""
    fpath = "./tests/test_data/chunks.csv"
    with open(fpath, "r", encoding="utf-8") as file:
        data = []
        reader = csv.reader(file, delimiter=";")
        for row in reader:
            data.append(row[2])  # column of chunk text
    data.pop(0)  # remove header row
    return data


@pytest.fixture(scope="function")
def topic_model():
    """TopicModel instance"""
    return TopicModel()


@pytest.mark.skip(reason="To be added.")
def test_load_nlp_pipe():
    """Unit test load_nlp_pipe."""
    pass


class TestGetChunk:
    """Tests for testing the get_chunk function."""

    test_data = [
        # cases of coordinating conjunction
        (
            [
                "This is my first car and I'm satisfied with it.",
                "I have two goldfish and a cat.",
            ],
            [0, 0, 1],
            [
                "I'm satisfied with it",
                "This is my first car and",
                "I have two goldfish and a cat",
            ],
        ),
        # cases of correlative conjunction
        (
            [
                "I will either go for a hike or stay home and watch TV.",
                "The kid was running as fast as us.",
            ],
            [0, 0, 1],
            [
                "stay home and watch TV",
                "I will either go for a hike or",
                "The kid was running as fast as us",
            ],
        ),
        # cases of subordinating conjunction
        (
            [
                "Unless we give him a ride, he will not be able to come.",
                "I'm not going to work because I am sick.",
            ],
            [0, 1],
            [
                "Unless we give him a ride he will not be able to come",
                "I'm not going to work because I am sick",
            ],
        ),
        # case of multiple sentences in one document
        (
            [
                "The suspension system is very poor. Driving on slightly uneven roads feels very bumpy."
            ],
            [0, 0],
            [
                "The suspension system is very poor",
                "Driving on slightly uneven roads feels very bumpy",
            ],
        ),
    ]

    def similarity(self, a: str, b: str) -> float:
        """Match two strings and return their similarity."""
        a = a.lower()
        b = b.lower()
        match = SequenceMatcher(a=a, b=b)
        return match.ratio()

    @pytest.mark.parametrize("docs, expected_arg_ids, expected_chunks", test_data)
    def test_get_chunks(self, docs, expected_arg_ids, expected_chunks):
        """Test of dealing with cases containing coordinating conjunctions. The result chunks might look a bit different from the expected chunks, since marks and the conjunctions themselves are not considered. Therefore, a similarity check is given to make sure that the most part fo the result chunks will meet the expectation."""
        arg_ids, chunks = get_chunk(docs=docs)

        assert arg_ids == expected_arg_ids
        for i, _ in enumerate(chunks):
            assert self.similarity(chunks[i], expected_chunks[i]) >= 0.9


def test_get_chunk_polarity_score():
    """Unit test get_chunk_polarity_score."""
    chunks = [
        "The shirt looks great!",  # extremely positive
        "The shirt looks awful!",  # extremely negative
        "Although I don't like it that much, I will still buy it since it's so cheap.",  # relatively positive
        "Although I like it so much, I won't buy it since it's too expensive.",  # relatively negative
        "That car is made in Japan.",  # neutral
    ]
    scores = get_chunk_polarity_score(chunks=chunks)

    assert scores[0] > 0.5
    assert scores[1] < -0.5
    assert 0 < scores[2] < 0.5
    assert -0.5 < scores[3] < 0
    assert scores[4] == 0


def test_get_chunk_rank():
    """Unit test get_chunk_rank."""
    arg_ids = [0, 0, 0, 0]
    embeds = np.array(
        [
            [0, 1],
            [0.2, 0.8],
            [0.4, 1.6],
            [1, 0],
        ]
    )
    ranks = get_chunk_rank(arg_ids=arg_ids, embeds=embeds)

    assert len(ranks) == 4
    assert ranks[1] == ranks[2] > ranks[0] > ranks[3]


def test_get_chunk_topic(mocker):
    """Unit test get_chunk_topic."""
    dummy_chunks = ["chunk1", "chunk2", "chunk3"]
    loc_str = "orangecontrib.argument.miner.chunker.TopicModel."

    mock_fit_transform_reduced = mocker.patch(
        loc_str + "fit_transform_reduced", return_value="dummy_topics"
    )
    mock_get_topic_table = mocker.patch(
        loc_str + "get_topic_table", return_value="dummy_table"
    )
    mock_get_doc_embeds = mocker.patch(
        loc_str + "get_doc_embeds", return_value="dummy_embeds"
    )

    topics, embeds, df_topics = get_chunk_topic(dummy_chunks)

    assert topics == "dummy_topics"
    assert embeds == "dummy_embeds"
    assert df_topics == "dummy_table"
    mock_fit_transform_reduced.assert_called_once_with(dummy_chunks)
    mock_get_topic_table.assert_called_once()
    mock_get_doc_embeds.assert_called_once()


def test_get_chunk_table():
    """Unit test get_chunk_table"""
    arg_ids = [0, 0, 1, 1]
    chunks = ["chunk0", "chunk1", "chunk2", "chunk3"]
    p_scores = [0.1, -0.5, 0, 0.3]
    topics = [0, 1, 1, 0]
    ranks = [0.2, 0.8, 0.5, 0.5]
    df_result = get_chunk_table(
        arg_ids=arg_ids, chunks=chunks, p_scores=p_scores, topics=topics, ranks=ranks
    )
    expected_cols = ["argument_id", "chunk", "polarity_score", "topic", "rank"]

    assert df_result.shape == (4, 5)
    assert all(col in df_result.columns for col in expected_cols)


@pytest.mark.skip(reason="To be added.")
def test_integrate_chunker():
    """Integration test functions of the chunker module."""
    pass


# TODO: switch to pytest-mock from unittest.mock
class TestTopicModel:
    """Tests of the TopicModel class."""

    def test_init_model(self, topic_model):
        """Unit test init_model."""
        topic_model.init_model()
        assert topic_model.model is not None

    def test_fit_transform_reduced(self, topic_model):
        """Unit test fit_transform_reduced."""
        size = 10
        chunks = ["This is dummy chunk%d." % i for i in range(size)]
        expected_topics = [-1] * 10
        topic_model.model = Mock()
        topic_model.model.fit_transform.return_value = (expected_topics, None)
        topic_model.model.reduce_outliers.return_value = expected_topics

        result = topic_model.fit_transform_reduced(chunks)

        assert result == expected_topics
        topic_model.model.fit_transform.assert_called_once_with(chunks)
        topic_model.model.reduce_outliers.assert_called_once_with(
            chunks, expected_topics, strategy="embeddings"
        )

    def test_get_topic_table(self, topic_model):
        """Unit test get_topic_table."""
        topic_model.model = Mock()
        topic_model.model.get_topic_info.return_value = Mock(
            rename=lambda columns: "dummy_string"
        )

        result = topic_model.get_topic_table()

        assert result == "dummy_string"
        topic_model.model.get_topic_info.assert_called_once()

    def test_get_doc_embeds(self, topic_model):
        """Unit test get_doc_embeds."""
        topic_model._rd_model = Mock(embedding_="dummy_string")

        result = topic_model.get_doc_embeds()

        assert result == "dummy_string"

    @pytest.mark.skip("To be added.")
    def test_integrate_topic_model(self):
        """Integration test of the Topic Model class."""
        pass
