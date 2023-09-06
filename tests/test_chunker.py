"""Tests of the chunker module"""
from difflib import SequenceMatcher

import pytest
import numpy as np

from orangecontrib.argument.miner.chunker import get_chunk, get_chunk_rank


class TestGetChunk:
    """Tests of the get_chunk function."""

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


def test_get_chunk_rank():
    """Test function get_chunk_rank."""
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
