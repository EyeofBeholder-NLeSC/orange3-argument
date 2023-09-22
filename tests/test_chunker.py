"""Tests of the chunker module"""
from difflib import SequenceMatcher

import pytest
import numpy as np
import pandas as pd

from orangearg.argument.miner.chunker import (
    load_nlp_pipe,
    get_chunk,
    get_chunk_polarity_score,
    get_chunk_rank,
    get_chunk_topic,
    get_chunk_table,
)
from .conftest import large_chunk_set, review_set, topic_model


def test_load_nlp_pipe(mocker):
    """Unit test load_nlp_pipe."""

    def check_download(name):
        """When executing effect as a callable, it will need to pass in all the args and kwargs of the mocked function, thus we have a `name` kwarg as a placeholder, otherwise won't work."""
        if not mock_spacy_cli_download.called:
            raise (OSError)
        else:
            return "dummy_model"

    mock_spacy_load = mocker.patch(
        "spacy.load",
        side_effect=check_download,
    )
    mock_spacy_cli_download = mocker.patch("spacy.cli.download")

    nlp = load_nlp_pipe(model_name="dummy_name")

    assert nlp == "dummy_model"
    mock_spacy_load.assert_any_call(name="dummy_name")
    mock_spacy_cli_download.assert_called_once_with(model="dummy_name")


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
    loc_str = "orangearg.argument.miner.chunker.TopicModel."

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
    assert sorted(df_result.columns) == sorted(expected_cols)


def test_integrate_chunker(review_set):
    """Integration test functions of the chunker module."""
    arg_ids, chunks = get_chunk(docs=review_set)
    p_scores = get_chunk_polarity_score(chunks=chunks)
    topics, embeds, df_topics = get_chunk_topic(chunks=chunks)
    ranks = get_chunk_rank(arg_ids=arg_ids, embeds=embeds)
    df_chunks = get_chunk_table(
        arg_ids=arg_ids,
        chunks=chunks,
        p_scores=p_scores,
        topics=topics,
        ranks=ranks,
    )

    assert len(set(arg_ids)) == len(review_set), [
        i for i in range(len(review_set)) if i not in arg_ids
    ]
    assert len(chunks) == len(p_scores) == len(topics) == len(embeds) == len(ranks)
    assert len(set(topics)) == df_topics.shape[0]
    assert all(-1.0 <= p <= 1.0 for p in p_scores)
    assert all(0.0 <= r <= 1.0 for r in ranks)
    assert isinstance(df_topics, pd.DataFrame)
    assert isinstance(df_chunks, pd.DataFrame)


class TestTopicModel:
    """Tests of the TopicModel class."""

    def test_init_model(self, topic_model):
        """Unit test init_model."""
        topic_model.init_model()
        assert topic_model.model is not None

    def test_fit_transform_reduced(self, mocker, topic_model):
        """Unit test fit_transform_reduced."""
        size = 10
        chunks = ["This is dummy chunk%d." % i for i in range(size)]
        expected_topics = [-1] * size
        mock_fit_transform = mocker.patch.object(
            topic_model.model,
            "fit_transform",
            return_value=(expected_topics, None),
        )
        mock_reduce_outliers = mocker.patch.object(
            topic_model.model,
            "reduce_outliers",
            side_effect=ValueError,
        )
        mock_update_topics = mocker.patch.object(topic_model.model, "update_topics")

        result = topic_model.fit_transform_reduced(chunks)

        assert result == expected_topics
        mock_fit_transform.assert_called_once_with(chunks)
        mock_reduce_outliers.assert_called_once_with(
            chunks, expected_topics, strategy="embeddings"
        )
        mock_update_topics.assert_called_once_with(chunks, topics=expected_topics)

    def test_get_topic_table(self, mocker, topic_model):
        """Unit test get_topic_table."""
        mock_get_topic_info = mocker.patch.object(
            topic_model.model,
            "get_topic_info",
            return_value=pd.DataFrame(
                {
                    "Representative_Docs": ["doc1"],
                    "Representation": [["keyword1", "keyword2"]],
                }
            ),
        )
        mock_rename = mocker.patch(
            "pandas.DataFrame.rename",
            return_value=pd.DataFrame({"keywords": [["keyword1", "keyword2"]]}),
        )

        result = topic_model.get_topic_table()

        assert isinstance(result, pd.DataFrame)
        assert result.compare(
            pd.DataFrame({"keywords": [("keyword1", "keyword2")]})
        ).empty
        mock_get_topic_info.assert_called_once()
        mock_rename.assert_called_once()

    def test_get_doc_embeds(self, mocker, topic_model):
        """Unit test get_doc_embeds."""
        topic_model._rd_model = mocker.Mock(embedding_=np.array([]))

        result = topic_model.get_doc_embeds()

        assert isinstance(result, np.ndarray)

    def test_integrate_topic_model(self, large_chunk_set, topic_model):
        """Integration test of the Topic Model class."""
        topics = topic_model.fit_transform_reduced(large_chunk_set)
        embeds = topic_model.get_doc_embeds()
        df_topics = topic_model.get_topic_table()

        assert len(large_chunk_set) == len(topics) == len(embeds)
        assert len(embeds[0]) == 5  # if size of embeddings aligns to n_component
        assert isinstance(df_topics, pd.DataFrame)
        assert df_topics.shape[0] > 2  # at least 2 clusters
        expected_cols = ["topic", "count", "name", "keywords"]
        assert sorted(df_topics.columns) == sorted(expected_cols)
