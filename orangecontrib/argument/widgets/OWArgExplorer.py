import numpy as np

from Orange.data import Table
from Orange.widgets.widget import Input
from Orange.widgets.visualize.utils.widget import OWDataProjectionWidget
from Orange.widgets.settings import SettingProvider
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
    
    def __init__(self):
        super().__init__()
        
        self.edge_data = None
        self.node_data = None
        self.positions = None
        
    def _add_controls(self):
        self.gui = OWPlotGUI(self)
        
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
        if self.positions is None:
            self.set_random_positions()
        
        self.closeContext()
        self.data = self.node_data
        self.valid_data = np.full(len(self.data), True, dtype=bool)
        self.openContext(self.data)
        
        self.graph.reset_graph()
        self.graph.update_coordinates()
        
    def set_positions(self):
        pass
            
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