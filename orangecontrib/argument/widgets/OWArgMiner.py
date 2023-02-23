from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget
from Orange.data.pandas_compat import table_from_frame, table_to_frame
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
    class Inputs:
        input_data = Input('Data', Table)
    
    class Outputs:
        edge_data = Output('Edge Data', Table)
        node_data = Output('Node Data', Table)
    
    def __init__(self):
        super().__init__()
       
        # GUI
        gui.button(
            widget=self.controlArea, 
            master=self, 
            label='Mine', 
            callback=self.commit,
        )
        
    @Inputs.input_data
    def set_data(self, data):
        self.input_data = data
         
    def commit(self):
        # argument mining
        progressbar = gui.ProgressBar(self, 100) 
        miner = ArgumentMiner(
            table_to_frame(self.input_data, include_metas=True))
        miner.load_nlp_pipeline() 
        progressbar.advance(10)
        miner.load_word_vector_model()
        progressbar.advance(80)
        miner.compute_ranks_and_readability()
        miner.compute_clusters_and_weights()
        miner.compute_edge_table()
        miner.compute_node_table()
        progressbar.finish()
        
        # send result to outputs
        self.Outputs.edge_data.send(table_from_frame(miner.df_edge))
        self.Outputs.node_data.send(table_from_frame(miner.df_node))


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview  # since Orange 3.20.0
    WidgetPreview(OWArgMiner).run()
