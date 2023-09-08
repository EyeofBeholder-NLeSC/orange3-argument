"""Test configuration."""
from pathlib import Path
import json

import pytest
import pandas as pd

from orangecontrib.argument.miner.chunker import TopicModel

CWD = Path("__file__").absolute().parent
TEST_DATA_FOLDER = CWD / "tests" / "test_data"


@pytest.fixture(scope="function")
def df_chunks():
    """Dataframe of chunks.

    The dataframe contains 5 columns that are:
    - argument_id (int): id of arguments that the chunk belongs to
    - chunk (str): chunk text
    - topic (int): topic index of a chunk
    - rank (float): pagerank of a chunk within the argument it belongs to
    - polarity_score (float): sentiment polarity score of a chunk
    """
    fpath = TEST_DATA_FOLDER / "chunks.csv"
    return pd.read_csv(fpath, delimiter=";", index_col=0)


@pytest.fixture(scope="function")
def df_arguments():
    """Dataframe of arguments.

    This dataframe contains 2 columns that are:
    - argument (str): argument text
    - score (int): overall score alongside of an argument, in range of [1, 5].
    """
    fpath = TEST_DATA_FOLDER / "reviews.json"
    with open(fpath, "r", encoding="utf-8") as file:
        data = []
        for obj in file:
            data.append(json.loads(obj))
    result = pd.json_normalize(data)
    return result.rename(columns={"reviewText": "argument", "overall": "score"})


@pytest.fixture(scope="function")
def large_chunk_set(df_chunks):
    """List of around 1200 chunk texts."""
    return df_chunks["chunk"].dropna().tolist()


@pytest.fixture(scope="function")
def review_set(df_arguments):
    """List of arund 370 review texts."""
    reviews = df_arguments["argument"].tolist()
    return [r for r in reviews if r is not None]


@pytest.fixture(scope="function")
def topic_model():
    """TopicModel instance"""
    return TopicModel()


@pytest.fixture(scope="function")
def mock_check_columns(mocker):
    return mocker.patch("orangecontrib.argument.miner.processor.check_columns")
