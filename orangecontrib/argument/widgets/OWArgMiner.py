"""Argument miner widget"""
from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget
from Orange.data.pandas_compat import table_from_frame, table_to_frame
from AnyQt.QtCore import Qt

from ..miner.miner import (
    select_by_topic,
    get_edges,
    get_edge_weights,
    get_edge_table,
    get_node_labels,
    get_node_table,
)
from ..miner.utilities import check_columns


class OWArgMiner(OWWidget):
    """Argument miner widget

    Given a set of arguments (read from URL), mine the attacking relationship and generate
    the corresponding edge and node table.
    """

    name = "Argument Miner"
    description = (
        "Mine argument set and create edge and node tables of their attacking network."
    )
    icon = "icons/OWArgMiner.svg"
    want_main_area = False

    # GUI variables
    selected_topic = Setting(0)

    class Inputs:
        argument_data = Input("Argument Data", Table)
        topic_data = Input("Topic Data", Table)

    class Outputs:
        edge_data = Output("Edge Data", Table)
        node_data = Output("Node Data", Table)

    def __init__(self):
        super().__init__()

        # GUI
        box_select_topic = gui.widgetBox(
            self.controlArea, orientation=Qt.Vertical, box="Select Topic"
        )
        self.combo_topic = gui.comboBox(
            box_select_topic,
            self,
            "selected_topic",
            label="Select a topic",
            sendSelectedValue=False,
            items=(),
        )
        gui.button(
            widget=self.controlArea,
            master=self,
            label="Mine",
            callback=self.commit,
        )

    @Inputs.argument_data
    def set_argument_data(self, data):
        self.df_arguments = table_to_frame(data, include_metas=True)
        expected_cols = ["argument", "score", "topics", "sentiment", "coherence"]
        check_columns(expected_cols=expected_cols, data=self.df_arguments)

    @Inputs.topic_data
    def set_topic_data(self, data):
        self.df_topics = table_to_frame(data, include_metas=True)
        check_columns(expected_cols=["topic"], data=self.df_topics)

        topics = self.df_topics["topic"].sort_values(ascending=True)
        topics = topics.drop(labels=[0]).reset_index(drop=True)
        topics = topics.astype(str).tolist()

        # update combobox
        self.combo_topic.clear()
        self.combo_topic.addItems(topics)
        self.selected_topic = 0

    def commit(self):
        """Callback: commit button."""
        arg_selection = select_by_topic(
            data=self.df_arguments, topic=self.selected_topic
        )
        edges = get_edges(data=arg_selection)
        weights = get_edge_weights(data=arg_selection, edges=edges)
        df_edges = get_edge_table(edges=edges, weights=weights)

        labels = get_node_labels(
            indices=arg_selection.index.tolist(),
            sources=df_edges["source"].tolist(),
            targets=df_edges["target"].tolist(),
        )
        df_nodes = get_node_table(
            arg_ids=arg_selection["argument_id"].tolist(),
            arguments=arg_selection["argument"].tolist(),
            scores=arg_selection["score"].tolist(),
            labels=labels,
        )

        self.Outputs.edge_data.send(table_from_frame(df_edges))
        self.Outputs.node_data.send(table_from_frame(df_nodes))
