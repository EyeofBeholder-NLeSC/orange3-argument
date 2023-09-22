"""Argument Chunker Widget"""
from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.widget import Input, Output, OWWidget
from Orange.data.pandas_compat import table_from_frame, table_to_frame

from ..miner.chunker import (
    get_chunk,
    get_chunk_polarity_score,
    get_chunk_rank,
    get_chunk_topic,
    get_chunk_table,
)
from ..miner.utilities import check_columns


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
        expected_cols = ["argument", "score"]
        check_columns(expected_cols=expected_cols, data=self.df_arguments)

    def chunk(self):
        """Call back: chunk arguments and output chunk and topic table."""
        arguments = self.df_arguments["argument"]
        arg_scores = self.df_arguments["score"]
        chunk_arg_ids, chunks = get_chunk(docs=arguments)
        chunk_p_scores = get_chunk_polarity_score(chunks=chunks)
        chunk_topics, chunk_embeds, df_topics = get_chunk_topic(chunks=chunks)
        chunk_ranks = get_chunk_rank(arg_ids=chunk_arg_ids, embeds=chunk_embeds)

        df_chunks = get_chunk_table(
            arg_ids=chunk_arg_ids,
            chunks=chunks,
            p_scores=chunk_p_scores,
            topics=chunk_topics,
            ranks=chunk_ranks,
        )

        # # list values in dataframe will encounter hashing issue.
        # # convert to string before output.
        # object_cols = df_topics.select_dtypes(include=[object]).columns
        # df_topics[object_cols] = df_topics[object_cols].astype(str)

        table_chunks = table_from_frame(df_chunks)
        table_topics = table_from_frame(df_topics)
        self.Outputs.chunk_data.send(table_chunks)
        self.Outputs.topic_data.send(table_topics)
