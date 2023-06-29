from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.widget import Input, Output, OWWidget
from Orange.data.pandas_compat import table_from_frame, table_to_frame
from orangecontrib.argument.miner.chunker import ArgumentChunker


class OWArgChunker(OWWidget):
    """Argument chunker widget.

    This widget accepts argument data with columns of text and overall score as input and output the chunks generated from arguments, as well as a table of topic information of chunks.
    """
    name = "Argument Chunker"
    description = "Chunk arguments and do topic modeling on top of that."
    icon = "icons/OWArgChunker.svg"
    want_main_area = False
    resizing_enabled = False
    
    class Inputs:
        input_data = Input("Data", Table)
        
    class Outputs:
        chunk_data = Output("Chunk Data", Table)
        topic_data = Output("Topic Data", Table)
    
    def __init__(self):
        super().__init__()
        self.df_arguments = None
        
        # GUI
        button_chunk = gui.button(self.controlArea, self, "Chunk", callback=self.chunk)
        
    @Inputs.input_data
    def set_input_data(self, data):
        self.df_arguments = table_to_frame(data, include_metas=True)
        
    def chunk(self):
        """Call back: chunk arguments and output chunk and topic table.
        """
        chunker = ArgumentChunker(self.df_arguments["argument"])
        df_chunks = chunker.get_chunk_table()
        df_topics = chunker.get_topic_table()
        table_chunks = table_from_frame(df_chunks)
        table_topics = table_from_frame(df_topics)
        self.Outputs.chunk_data.send(table_chunks)
        self.Outputs.topic_data.send(table_topics)
        
if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    from orangecontrib.argument.miner.reader import read_json_file
   
    fpath = "./example/data/data_processed_1prod_full.json" 
    df_arguments = read_json_file(fpath)
    df_arguments = df_arguments[["reviewText", "overall"]]
    df_arguments = df_arguments.rename(columns={
        "reviewText": "argument", 
        "overall": "score"
    })
    table_arguments = table_from_frame(df_arguments)
    WidgetPreview(OWArgChunker).run(input_data=table_arguments)
    