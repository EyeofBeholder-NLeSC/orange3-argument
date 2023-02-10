from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWBaseWidget, Output, OWWidget
from Orange.data.pandas_compat import table_from_frame
from orangecontrib.argument.miner.miner import ArgumentMiner 


class OWArgMiner(OWWidget):
    """Argument miner widget
    
    Given a set of arguments (read from URL), mine the attacking relationship and generate
    the corresponding edge and node table.
    """
    
    name = "Argument Miner"
    description = "Mine argument set and create edge and node tables of their attacking network."
    icon = "icons/OWArgMiner.svg"
    
    want_main_area = False
    
    # GUI variables
    input_url = Setting('')
    
    class Outputs:
        edge_data = Output('Edge Data', Table)
        node_data = Output('Node Data', Table)
    
    def __init__(self):
        super().__init__()
       
        # GUI
        gui.lineEdit(
            widget=self.controlArea, 
            master=self, 
            value='input_url', 
            label='Input URL',
        )
        gui.button(
            widget=self.controlArea, 
            master=self, 
            label='Mine', 
            callback=self.commit,
        )
        
    def commit(self):
        # argument mining
        miner = ArgumentMiner(self.input_url)
        miner.load_nlp_pipeline() 
        miner.load_word_vector_model()
        miner.compute_ranks_and_readability()
        miner.compute_clusters_and_weights()
        miner.compute_edge_table()
        miner.compute_node_table()
        
        # send result to outputs
        self.Outputs.edge_data.send(table_from_frame(miner.df_edge))
        self.Outputs.node_data.send(table_from_frame(miner.df_node))


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview  # since Orange 3.20.0
    WidgetPreview(OWArgMiner).run()
