"""Tests of the chunker module.

Tests focus on the most critical functions, which are `get_chunk` and `get_chunk_rank`. The reason why other classes and methods in the module are not tested is either because they are completely/mostly calling the classes and methods of other mature Python libraries (such as BERTopic and TextBlob), and the code logic is clear, or because these methods are not critical methods.
"""
import pytest

from orangecontrib.argument.miner.chunker import get_chunk, get_chunk_rank


class TestGetChunk:
    """Tests of the get_chunk function."""

    test_data = {
        "coordinating_conj": [
            "This is my first car and I'm satisfied with it.",
            "I have two goldfish and a cat.",
        ],
        "correlative_conj": [
            "I will either go for a hike or stay home and watch TV.",
            "The kid was running as fast as us.",
        ],
        "subordinating_conj": [
            "Unless we give him a ride, he will not be able to come.",
            "I'm not going to work because I am sick.",
        ],
        "multi_sent": [
            "The suspension system is very poor. Driving on slightly uneven roads feels very bumpy."
        ],
    }

    @pytest.mark.parametrize("docs", [test_data["coordinating_conj"]])
    def test_coordinating_conj(self, docs):
        """Test of dealing with cases containing coordinating conjunctions. The first sentence should be divide into two chunks, while the second should remain as a single chunk."""
        arg_ids, chunks = get_chunk(docs=docs)
        print(arg_ids)
        print(chunks)
        assert 0

    # TODO: finish up the rest of the tests.
