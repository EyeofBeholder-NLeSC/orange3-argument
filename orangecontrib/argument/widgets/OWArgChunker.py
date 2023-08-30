"""Argument Chunker Widget"""


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

    class Inputs:
        input_data = Input("Data", Table)

    class Outputs:
        chunk_data = Output("Chunk Data", Table)
        topic_data = Output("Topic Data", Table)

    def __init__(self):
        super().__init__()
        self.df_arguments = None

        # GUI
        gui.button(self.controlArea, self, "Chunk", callback=self.chunk)

    @Inputs.input_data
    def set_input_data(self, data):
        self.df_arguments = table_to_frame(data, include_metas=True)

    def chunk(self):
        """Call back: chunk arguments and output chunk and topic table."""
        chunker = ArgumentChunker(self.df_arguments["argument"])
        df_chunks = chunker.get_chunk_table()
        df_topics = chunker.get_topic_table()

        # list values in dataframe will encounter hashing issue.
        # convert to string before output.
        object_cols = df_topics.select_dtypes(include=[object]).columns
        df_topics[object_cols] = df_topics[object_cols].astype(str)

        table_chunks = table_from_frame(df_chunks)
        table_topics = table_from_frame(df_topics)
        self.Outputs.chunk_data.send(table_chunks)
        self.Outputs.topic_data.send(table_topics)
