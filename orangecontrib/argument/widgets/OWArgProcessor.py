"""Argument processor widget"""
from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.widget import Input, Output, OWWidget
from Orange.data.pandas_compat import table_from_frame, table_to_frame

from ..miner.processor import (
    get_argument_topics,
    get_argument_coherence,
    get_argument_sentiment,
    update_argument_table,
)
from ..miner.utilities import check_columns


class OWArgProcessor(OWWidget):
    """Argument processor widget.

    Merge chunks back into arguments with the associated meta information, and compute
    other measures like readibility, sentiment, and coherence of arguments.
    """

    name = "Argument Processor"
    description = "Process input argument table and prepare it for the mining step."
    icon = "icons/OWArgProcessor.svg"
    want_main_area = False

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
        gui.button(self.controlArea, self, label="Process", callback=self.process)

    @Inputs.argument_data
    def set_argument_data(self, data):
        self.df_arguments = table_to_frame(data, include_metas=True)
        expected_cols = ["argument", "score"]
        check_columns(expected_cols=expected_cols, data=self.df_arguments)

    @Inputs.chunk_data
    def set_chunk_data(self, data):
        self.df_chunks = table_to_frame(data, include_metas=True)
        expected_cols = ["argument_id", "chunk", "polarity_score", "topic", "rank"]
        check_columns(expected_cols=expected_cols, data=self.df_chunks)

    def process(self):
        """Call back: merge chunks back to arguments and compute all the measures."""
        arg_topics = get_argument_topics(
            arg_ids=self.df_chunks["argument_id"], topics=self.df_chunks["topic"]
        )
        arg_sentiments = get_argument_sentiment(
            arg_ids=self.df_chunks["argument_id"],
            ranks=self.df_chunks["rank"],
            p_scores=self.df_chunks["polarity_score"],
        )
        arg_coherences = get_argument_coherence(
            scores=self.df_arguments["score"], sentiments=arg_sentiments
        )

        condition = (
            len(arg_topics)
            == len(arg_sentiments)
            == len(arg_coherences)
            == self.df_arguments.shape[0]
        )
        if not condition:
            raise ValueError(
                f"Size of the processing result not aling to the given argument table: {len(arg_topics)}, {len(arg_sentiments)}, {len(arg_coherences)}, {self.df_arguments.shape[0]}"
            )

        df_arguments_processed = update_argument_table(
            df_arguments=self.df_arguments,
            topics=arg_topics,
            sentiments=arg_sentiments,
            coherences=arg_coherences,
        )

        table_arguments = table_from_frame(df_arguments_processed)
        self.Outputs.argument_data.send(table_arguments)
