"""Test the miner module."""
import pytest
import pandas as pd

from orangearg.argument.miner.miner import (
    select_by_topic,
    get_edges,
    get_edge_weights,
    get_edge_table,
    get_node_labels,
    get_node_table,
)
from .conftest import dummy_argument_selection, dummy_edge_data, df_arguments_processed


@pytest.mark.parametrize(
    "dummy_data, dummy_topic, expected_result",
    [
        (
            pd.DataFrame({"topics": [(0), (1, 2)]}),
            0,
            pd.DataFrame({"topics": [(0)], "argument_id": [0]}),
        ),
        (
            pd.DataFrame({"topics": [(), (1, 2)]}),
            0,
            pd.DataFrame({"topics": [], "argument_id": []}),
        ),
        (
            pd.DataFrame({"topics": ["(0)", "(1, 2)"]}),
            0,
            pd.DataFrame({"topics": ["(0)"], "argument_id": [0]}),
        ),
    ],
)
def test_select_by_topic(dummy_data, dummy_topic, expected_result):
    """Unit test function selection_by_topic."""
    result = select_by_topic(data=dummy_data, topic=dummy_topic)
    assert isinstance(result, pd.DataFrame)
    assert result.compare(expected_result).empty


def test_get_edges(dummy_argument_selection):
    """Unit test function get_edges."""
    expected_result = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 3), (1, 4), (2, 3), (2, 4)]

    result = get_edges(data=dummy_argument_selection)

    assert result == expected_result


def test_get_edge_weights(dummy_argument_selection):
    """Unit test function get_edge_weights."""
    edges = get_edges(dummy_argument_selection)
    expected_result = [-0.2, 0.2, 0.25, -0.6, 0.45, -0.4, 0.05, -0.8]

    result = get_edge_weights(data=dummy_argument_selection, edges=edges)

    assert result == expected_result


def test_get_edge_table(dummy_argument_selection, dummy_edge_data):
    """Unit test function get_edge_table.

    Edge nr. 0, 3, 5, 7 have their source and target swapped, as their weights are negative.
    """
    edges = get_edges(dummy_argument_selection)
    weights = get_edge_weights(data=dummy_argument_selection, edges=edges)

    result = get_edge_table(edges=edges, weights=weights)

    assert result.compare(dummy_edge_data).empty


def test_get_node_labels(dummy_argument_selection, dummy_edge_data):
    """Unit test function get_node_labels.

    Argument 3 is supportive because all its attackers are attacked by some other arguments, while argument 4 is supportive because no other arguments attack it.
    """
    dummy_indices = dummy_argument_selection.index.tolist()
    dummy_sources = dummy_edge_data["source"].tolist()
    dummy_targets = dummy_edge_data["target"].tolist()
    expected_result = ["defeated", "defeated", "defeated", "supportive", "supportive"]

    result = get_node_labels(
        indices=dummy_indices, sources=dummy_sources, targets=dummy_targets
    )

    assert result == expected_result


def test_get_node_table(dummy_argument_selection, dummy_edge_data):
    """Unit test function_get_node_table."""
    dummy_indices = dummy_argument_selection.index.tolist()
    dummy_sources = dummy_edge_data["source"].tolist()
    dummy_targets = dummy_edge_data["target"].tolist()
    dummy_arg_ids = dummy_argument_selection["argument_id"].tolist()
    dummy_arguments = dummy_argument_selection["argument"].tolist()
    dummy_scores = dummy_argument_selection["score"].tolist()
    dummy_labels = get_node_labels(
        indices=dummy_indices, sources=dummy_sources, targets=dummy_targets
    )
    expected_result = pd.DataFrame(
        {
            "argument_id": [4, 7, 13, 29, 33],
            "argument": ["arg1", "arg2", "arg3", "arg4", "arg5"],
            "score": [1, 2, 2, 3, 3],
            "label": ["defeated", "defeated", "defeated", "supportive", "supportive"],
        }
    )

    result = get_node_table(
        arg_ids=dummy_arg_ids,
        arguments=dummy_arguments,
        scores=dummy_scores,
        labels=dummy_labels,
    )

    assert result.compare(expected_result).empty


def test_integrate_miner(df_arguments_processed):
    """Integrate test the miner module."""
    selection = select_by_topic(data=df_arguments_processed, topic=22)
    assert selection.shape[0] == 19

    edges = get_edges(data=selection)
    weights = get_edge_weights(data=selection, edges=edges)
    df_edges = get_edge_table(edges=edges, weights=weights)
    assert df_edges.shape[0] == 50

    indices = selection.index.tolist()
    sources = df_edges["source"].tolist()
    targets = df_edges["target"].tolist()
    labels = get_node_labels(indices=indices, sources=sources, targets=targets)
    assert labels[5] == labels[6] == labels[14] == "defeated"

    arg_ids = selection["argument_id"].tolist()
    arguments = selection["argument"].tolist()
    scores = selection["score"].tolist()
    df_nodes = get_node_table(
        arg_ids=arg_ids, arguments=arguments, scores=scores, labels=labels
    )
