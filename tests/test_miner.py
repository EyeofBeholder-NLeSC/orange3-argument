import pytest
import pandas as pd
import numpy as np
from miner import ArgumentMiner


# Test select_by_topic method
def test_select_by_topic():
    df_arguments = pd.DataFrame({'topics': [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]],
                                 'argument_id': [1, 2, 3, 4]})
    miner = ArgumentMiner(df_arguments)
    result = miner.select_by_topic(3)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert result['topics'].tolist() == [[1, 2, 3], [2, 3, 4], [3, 4, 5]]

# Test get_edge_table method
def test_get_edge_table():
    df_selection = pd.DataFrame({
        "argument_id": [1, 2, 3, 4], 
        "score": [3, 3, 4, 5],
        "coherence": [0.3, 0.5, 0.7, 0.4]
    })
    miner = ArgumentMiner(df_arguments=None)
    result = miner.get_edge_table(df_selection)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 5
    assert pd.Series(["weight", "source", "target"]).isin(result.columns).all()
    print(result)
    assert result['source'].tolist() == [3, 4, 3, 2, 3]
    assert result['target'].tolist() == [1, 1, 2, 4, 4]

# Test get_node_table method
def test_get_node_table():
    df_edges = pd.DataFrame({'source': [3, 4, 3, 2, 3],
                             'target': [1, 1, 2, 4, 4]})
    df_nodes = pd.DataFrame({'argument_id': [1, 2, 3, 4],
                             'text': ['Argument 1', 'Argument 2', 'Argument 3', 'Argument 4']})
    miner = ArgumentMiner(df_arguments=None)
    result = miner.get_node_table(df_edges, df_nodes)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4
    assert pd.Series(['argument_id', 'text', 'label']).isin(result.columns).all()
    assert result['label'].tolist() == ['defeated', 'defeated', 'supportive', 'defeated']

