import pytest
import pandas as pd
from processor import ArgumentProcessor

@pytest.fixture(scope="session")
def df_arguments():
    yield pd.read_csv("./tests/test_data/df_arguments.csv", sep=";")
    
@pytest.fixture(scope="session")
def df_chunks():
    yield pd.read_csv("./tests/test_data/df_chunks.csv", sep=";")

# Test argument readability computation
def test_argument_readability(df_arguments):
    processor = ArgumentProcessor(df_arguments)
    processor.argument_readability()
    assert 'readability' in processor.df_arguments.columns

# Test argument topics computation
def test_argument_topics(df_arguments, df_chunks):
    processor = ArgumentProcessor(df_arguments)
    processor.argument_topics(df_chunks)
    assert 'topics' in processor.df_arguments.columns
    assert isinstance(processor.df_arguments.loc[0]["topics"], list)

# Test argument sentiment computation
def test_argument_sentiment(df_arguments, df_chunks):
    processor = ArgumentProcessor(df_arguments)
    processor.argument_sentiment(df_chunks)
    assert 'sentiment' in processor.df_arguments.columns
    assert processor.df_arguments["sentiment"].max() <= 1
    assert processor.df_arguments["sentiment"].min() >= 0
    
    # test particular cases
    assert processor.df_arguments.loc[0]["sentiment"] < 0.5
    assert processor.df_arguments.loc[1]["sentiment"] > 0.5
    assert processor.df_arguments.loc[2]["sentiment"] == 0.5

# Test argument coherence computation
def test_argument_coherence(df_arguments, df_chunks):
    processor = ArgumentProcessor(df_arguments)
    processor.argument_sentiment(df_chunks)
    processor.argument_coherence(df_chunks)
    assert 'coherence' in processor.df_arguments.columns
    assert processor.df_arguments["coherence"].max() <= 1
    assert processor.df_arguments["coherence"].min() >= 0
    
    # test particular cases
    assert processor.df_arguments.loc[0]["coherence"] > \
        processor.df_arguments.loc[1]["coherence"]
    assert processor.df_arguments.loc[1]["coherence"] > \
        processor.df_arguments.loc[2]["coherence"]

# Test get_argument_table method
def test_get_argument_table(df_arguments, df_chunks):
    processor = ArgumentProcessor(df_arguments)
    result = processor.get_argument_table(df_chunks)
    assert isinstance(result, pd.DataFrame)
    assert 'readability' in result.columns
    assert 'topics' in result.columns
    assert 'sentiment' in result.columns
    assert 'coherence' in result.columns

