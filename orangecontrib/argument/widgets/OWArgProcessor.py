from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.widget import Input, Output, OWWidget
from Orange.data.pandas_compat import table_from_frame, table_to_frame
from orangecontrib.argument.miner.processor import ArgumentProcessor


class OWArgProcessor(OWWidget):
    """Argument processor widget.
    
    Merge chunks back into arguments with the associated meta information, and compute
    other measures like readibility, sentiment, and coherence of arguments. 
    """
    name = 'Argument Processor'
    description = 'Process input argument table and prepare it for the mining step.'
    icon = 'icons/OWArgProcessor.svg'
    want_main_area = False
    resizing_enabled = False
    
    # GUI variables
    
    class Inputs:
        argument_data = Input("Argument Data", Table)
        chunk_data = Input("Chunk Data", Table)
        
    class Outputs:
        argument_data = Output("Argument Data", Table) 
       
    def __init__(self):
        super().__init__() 
        self.df_arguments = None
        self.df_chunks = None
        
        # GUI
        gui.button(self.controlArea, self, label='Process', callback=self.process)
    
    @Inputs.argument_data
    def set_argument_data(self, data):
        self.df_arguments = table_to_frame(data, include_metas=True)   
        
    @Inputs.chunk_data
    def set_chunk_data(self, data):
        self.df_chunks = table_to_frame(data, include_metas=True)
     
    def process(self):
        """Call back: merge chunks back to arguments and compute all the measures.
        """ 
        processor = ArgumentProcessor(self.df_arguments)
        self.df_arguments = processor.get_argument_table(self.df_chunks)
        table_arguments = table_from_frame(self.df_arguments)
        self.Outputs.output_data.send(table_arguments)
       

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    from orangecontrib.argument.miner.reader import read_json_file
    from orangecontrib.argument.miner.chunker import ArgumentChunker
    
    fpath = "./example/data/data_processed_1prod_full.json"
    df_arguments = read_json_file(fpath)
    df_arguments = df_arguments[["reviewText", "overall"]]
    df_arguments = df_arguments.rename(columns={
        "reviewText": "argument", 
        "overall": "score"
    })
    chunker = ArgumentChunker(df_arguments["argument"])
    df_chunks = chunker.get_chunk_table()
    df_topics = chunker.get_topic_table()
    table_chunks = table_from_frame(df_chunks)
    table_arguments = table_from_frame(df_arguments)
    WidgetPreview(OWArgProcessor).run(argument_data=df_arguments, chunk_data=df_chunks)