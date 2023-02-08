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

from orangecontrib.argument.graph.graphview import GraphView

class OWArgExplorer(OWDataProjectionWidget):
    name = 'Argument Explorer'
    description = 'Visually explore arguments in their attacking network.'
    icon = 'icons/OWArgExplorer.svg'
    
    class Inputs:
        edge_data = Input('Edge Data', Table)
        node_data = Input('Node Data', Table)
        
    GRAPH_CLASS = GraphView # borrowed from Orange3-network add-on
    graph = SettingProvider(GraphView) 
    
    node_sparsity = Setting(5)
    
    def __init__(self):
        super().__init__()
        
        self.edge_data = None
        self.node_data = None
        self.positions = None
        
    def _add_controls(self):
        self.gui = OWPlotGUI(self)
        layout = gui.vBox(self.controlArea, box='Layout') 
        gui.hSlider(layout, self, "node_sparsity", 
                    minValue=0, maxValue=10, intOnly=False, 
                    label="Node sparsity", orientation=Qt.Horizontal,
                    callback_finished=self.relayout) 
        
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
        if self.node_data is None or self.edge_data is None:
            return
        self.set_positions()
        self.closeContext()
        self.data = self.node_data
        self.valid_data = np.full(len(self.data), True, dtype=bool)
        self.openContext(self.data)
        self.graph.reset_graph()
        
    def set_positions(self, layout="default"):
        """set coordinates of nodes to self.positions.

        Args:
            layout (str, optional): name of layout. Defaults to "sfdp".
        """
        df_edge = table_to_frame(self.edge_data)
        df_node = table_to_frame(self.node_data)
        
        # normalize weights of edges
        # df_edge['weight'] = (df_edge['weight'] - df_edge['weight'].min()) \
            # / (df_edge['weight'].max() - df_edge['weight'].min())
        df_edge['weight'] /= df_edge['weight'].max()
        
        G = nx.from_pandas_edgelist(
            df_edge, 
            source='source', target='target', edge_attr=['weight'], 
            create_using=nx.DiGraph()) 
        
        # in case arguments not appear in the attacking network
        if len(G.nodes) < df_node.shape[0]:
            remain_nodes = df_node.iloc[~df_node.index.isin(G.nodes)]
            G.add_nodes_from(remain_nodes.index.tolist())
       
        if layout == 'default':
            spasity = (self.node_sparsity + 1) / 11.0
            pos_dict = nx.spring_layout(G, k=spasity, seed=10)
            print(pos_dict)
       
        self.positions = []
        for i in sorted(pos_dict.keys()):
            self.positions.append(pos_dict[i])  
        self.positions = np.array([*self.positions])
        
    def set_random_positions(self):
        if self.node_data is not None:
            num_nodes = len(self.node_data)
            self.positions = np.random.uniform(size=(num_nodes, 2))
            
    def get_embedding(self):
        return self.positions
    
    def get_edges(self):
        return table_to_frame(self.edge_data)
    
    def get_marked_nodes(self):
        return None
    
    def get_node_labels(self):
        return table_to_frame(self.node_data)['label']