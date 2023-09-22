"""Argument Explorer Widget"""
import numpy as np
import networkx as nx
from AnyQt.QtCore import Qt
from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.widget import Input
from Orange.widgets.visualize.utils.widget import OWDataProjectionWidget
from Orange.widgets.settings import SettingProvider, Setting
from Orange.widgets.utils.plot import OWPlotGUI
from Orange.data.pandas_compat import table_to_frame

from ..graph.graphview import GraphView

GRAPH_LAYOUT = ("spring", "multipartite", "kamada kawai", "spectral")


class OWArgExplorer(OWDataProjectionWidget):
    name = "Argument Explorer"
    description = "Visually explore arguments in their attacking network."
    icon = "icons/OWArgExplorer.svg"

    class Inputs:
        edge_data = Input("Edge Data", Table)
        node_data = Input("Node Data", Table)

    GRAPH_CLASS = GraphView
    graph = SettingProvider(GraphView)

    node_sparsity = Setting(5)
    graph_layout = Setting(
        GRAPH_LAYOUT[0]
    )  # comboBox widget returns index of the selection

    def __init__(self):
        super().__init__()

        self.edge_data = None
        self.node_data = None
        self.positions = None

    def _add_controls(self):
        self.gui = OWPlotGUI(self)
        layout = gui.vBox(self.controlArea, box="Layout")
        gui.comboBox(
            layout,
            self,
            "graph_layout",
            label="Graph layout",
            sendSelectedValue=True,
            items=GRAPH_LAYOUT,
            callback=self.relayout,
        )
        self.sparsity_control = gui.hSlider(
            layout,
            self,
            "node_sparsity",
            minValue=0,
            maxValue=10,
            intOnly=False,
            label="Node sparsity",
            orientation=Qt.Horizontal,
            callback_finished=self.relayout,
        )

    @Inputs.edge_data
    def set_edge_data(self, data):
        self.edge_data = data

    @Inputs.node_data
    def set_node_data(self, data):
        self.node_data = data

    def handleNewSignals(self):
        self.AlphaValue = 0
        self.relayout()

    def relayout(self):
        """recompute positions of nodes and reset the graph"""
        if self.node_data is None or self.edge_data is None:
            return

        self.sparsity_control.setEnabled(self.graph_layout == GRAPH_LAYOUT[0])
        self.set_positions()
        self.closeContext()
        self.data = self.node_data
        self.valid_data = np.full(len(self.data), True, dtype=bool)
        self.openContext(self.data)
        self.graph.reset_graph()

    def set_positions(self):
        """set coordinates of nodes to self.positions.

        Args:
            layout (str, optional): name of layout. Defaults to "sfdp".
        """
        df_edge = table_to_frame(self.edge_data)
        df_node = table_to_frame(self.node_data)

        G = nx.from_pandas_edgelist(
            df_edge,
            source="source",
            target="target",
            edge_attr=["weight"],
            create_using=nx.DiGraph(),
        )
        node_attrs = {i: {"subset": df_node["label"][i]} for i in G.nodes}
        nx.set_node_attributes(G, node_attrs)

        # in case arguments not appear in the attacking network
        if len(G.nodes) < df_node.shape[0]:
            remain_nodes = df_node.iloc[~df_node.index.isin(G.nodes)]
            G.add_nodes_from(remain_nodes.index.tolist())

        if self.graph_layout == GRAPH_LAYOUT[0]:
            spasity = (self.node_sparsity + 1) / 11.0
            pos_dict = nx.spring_layout(G, k=spasity, seed=10)
        elif self.graph_layout == GRAPH_LAYOUT[1]:
            pos_dict = nx.multipartite_layout(G)
        elif self.graph_layout == GRAPH_LAYOUT[2]:
            pos_dict = nx.kamada_kawai_layout(G)
        elif self.graph_layout == GRAPH_LAYOUT[3]:
            pos_dict = nx.spectral_layout(G)

        self.positions = []
        for i in sorted(pos_dict.keys()):
            self.positions.append(pos_dict[i])
        self.positions = np.array([*self.positions])

    def get_embedding(self):
        return self.positions

    def get_edges(self):
        return table_to_frame(self.edge_data)

    def get_marked_nodes(self):
        return None

    def get_node_labels(self):
        return table_to_frame(self.node_data)["label"]

    def selection_changed(self):
        super().selection_changed()
        self.graph.update_edges()
