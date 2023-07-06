from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget
from Orange.data.pandas_compat import table_from_frame, table_to_frame
from orangecontrib.argument.miner.miner import ArgumentMiner 
from AnyQt.QtCore import Qt


class OWArgMiner(OWWidget):
    """Argument miner widget
    
    Given a set of arguments (read from URL), mine the attacking relationship and generate
    the corresponding edge and node table.
    """
    
    name = "Argument Miner"
    description = "Mine argument set and create edge and node tables of their attacking network."
    icon = "icons/OWArgMiner.svg"
    resizing_enabled = False 
    want_main_area = False
    
    # GUI variables
    selected_topic = Setting(0)
   
    class Inputs:
        argument_data = Input('Argument Data', Table)
        topic_data = Input('Topic Data', Table) 
    
    class Outputs:
        edge_data = Output('Edge Data', Table)
        node_data = Output('Node Data', Table)
    
    def __init__(self):
        super().__init__()
        
        # GUI
        box_select_topic = gui.widgetBox(self.controlArea, 
                                         orientation=Qt.Vertical, 
                                         box="Select Topic")
        self.combo_topic = gui.comboBox(box_select_topic, self, 
                                        'selected_topic', label="Select a topic", 
                                        sendSelectedValue=False, items=())
        gui.button(
            widget=self.controlArea, 
            master=self, 
            label='Mine', 
            callback=self.commit,
        )
        
    @Inputs.argument_data
    def set_argument_data(self, data):
        self.df_arguments = table_to_frame(data, include_metas=True)
        
    @Inputs.topic_data
    def set_topic_data(self, data):
        self.df_topics = table_to_frame(data, include_metas=True)
        topics = self.df_topics["topic"].sort_values(ascending=True)
        topics = topics.drop(labels=[0]).reset_index(drop=True)
        topics = topics.astype(str).tolist()
        
        # update combobox
        self.combo_topic.clear()
        self.combo_topic.addItems(topics)
        self.selected_topic = 0
        
         
    def commit(self):
        # argument mining
        progressbar = gui.ProgressBar(self, 100) 
        progressbar.advance(10)
        miner = ArgumentMiner(self.df_arguments)
        progressbar.advance(20)
        df_nodes = miner.select_by_topic(int(self.selected_topic))
        df_edges = miner.get_edge_table(df_nodes)
        df_nodes = miner.get_node_table(df_edges, df_nodes)
        progressbar.finish()
        
        # send result to outputs
        self.Outputs.edge_data.send(table_from_frame(df_edges))
        self.Outputs.node_data.send(table_from_frame(df_nodes))