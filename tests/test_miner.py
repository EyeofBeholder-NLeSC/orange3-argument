"""Test the miner module."""
import pytest
from pytest import approx
import pandas as pd

from orangecontrib.argument.miner.miner import (
    select_by_topic,
    get_edges,
    get_edge_weights,
)

from .conftest import dummy_argument_data


@pytest.mark.parametrize(
    "dummy_data, dummy_topic, expected_result",
    [
        (
            pd.DataFrame({"topics": [(0), (1, 2)]}),
            0,
            pd.DataFrame({"topics": [(0)]}),
        ),
        (
            pd.DataFrame({"topics": [(), (1, 2)]}),
            0,
            pd.DataFrame({"topics": []}),
        ),
        (
            pd.DataFrame({"topics": ["(0)", "(1, 2)"]}),
            0,
            pd.DataFrame({"topics": ["(0)"]}),
        ),
    ],
)
def test_select_by_topic(dummy_data, dummy_topic, expected_result):
    """Unit test function selection_by_topic."""
    result = select_by_topic(data=dummy_data, topic=dummy_topic)
    assert isinstance(result, pd.DataFrame)
    assert result.compare(expected_result).empty


def test_get_edges(dummy_argument_data):
    """Unit test function get_edges."""
    expected_result = [(0, 1), (0, 2), (0, 3), (1, 3), (2, 3)]

    result = get_edges(data=dummy_argument_data)

    assert result == expected_result


def test_get_edge_weights(dummy_argument_data):
    """Unit test function get_edge_weights."""
    edges = [(0, 1), (0, 2), (0, 3), (1, 3), (2, 3)]
    expected_result = [-0.2, 0.2, -0.6, -0.4, -0.8]

    result = get_edge_weights(data=dummy_argument_data, edges=edges)

    assert result == approx(expected_result, 0.01)  # float subtraction
