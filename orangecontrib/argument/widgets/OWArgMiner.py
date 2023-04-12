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
   
    # # GUI variables 
    # col_argument = Setting("")
    # col_score = Setting("")
    
    class Inputs:
        input_data = Input('Data', Table)
    
    class Outputs:
        edge_data = Output('Edge Data', Table)
        node_data = Output('Node Data', Table)
    
    def __init__(self):
        super().__init__()
       
        # GUI
        # select_columns = gui.vBox(self.controlArea, box='Select Columns')
        # self.select_col_arg = gui.comboBox(select_columns, self, 'col_argument', 
        #              label='Column of Argument Text', 
        #              sendSelectedValue=True,
        #              items=())
        # self.select_col_score = gui.comboBox(select_columns, self, 'col_score', 
        #              label="Column of Argument Score", 
        #              sendSelectedValue=True,
        #              items=())
        gui.button(
            widget=self.controlArea, 
            master=self, 
            label='Mine', 
            callback=self.commit,
        )
        
    @Inputs.input_data
    def set_input_data(self, data):
        self.input_data = data
        # df = table_to_frame(data, include_metas=True)
        # items = list(df.columns)
        # self.select_col_arg.clear()
        # self.select_col_arg.addItems(items)
        # self.col_argument = items[0]
        # self.select_col_score.clear()
        # self.select_col_score.addItems(items)
        # self.col_score = items[0]
         
    def commit(self):
        # argument mining
        progressbar = gui.ProgressBar(self, 100) 
        miner = ArgumentMiner(
            table_to_frame(self.input_data, include_metas=True))
        # miner.rename_column(self.col_argument, 'argument')
        # miner.rename_column(self.col_score, 'score')
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