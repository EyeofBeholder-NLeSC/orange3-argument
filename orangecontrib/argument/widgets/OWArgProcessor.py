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

    @Inputs.chunk_data
    def set_chunk_data(self, data):
        self.df_chunks = table_to_frame(data, include_metas=True)

    def process(self):
        """Call back: merge chunks back to arguments and compute all the measures."""
        processor = ArgumentProcessor(self.df_arguments)
        self.df_arguments = processor.get_argument_table(self.df_chunks)

        # deal with the type error when hashing list values in df
        object_cols = self.df_arguments.select_dtypes(include=[object]).columns
        self.df_arguments[object_cols] = self.df_arguments[object_cols].astype(str)

        table_arguments = table_from_frame(self.df_arguments)
        self.Outputs.argument_data.send(table_arguments)
