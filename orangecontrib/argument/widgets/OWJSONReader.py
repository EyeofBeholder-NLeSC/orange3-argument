"""JSON Reader widget."""
import os

from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Output, OWWidget
from Orange.data.pandas_compat import table_from_frame
from AnyQt.QtWidgets import QFileDialog
from AnyQt.QtCore import Qt

from ..miner.reader import read_json_file


class OWJSONReader(OWWidget):
    """Json file reader widget

    This widget reads a local JSON file and output the data as table.
    """

    name = "JSON Reader"
    description = "Read a local JSON file and output the data as table."
    icon = "icons/OWJSONReader.svg"
    want_main_area = False

    # GUI variables
    fpath = Setting("")

    class Outputs:
        output_data = Output("Data", Table)

    def __init__(self):
        super().__init__()

        # GUI
        box_select_file = gui.widgetBox(
            self.controlArea, orientation=Qt.Horizontal, box="Select File"
        )
        gui.button(box_select_file, self, "...", callback=self.browse_file)
        self.label_path = gui.label(
            box_select_file, self, "Path: %(fpath)s", labelWidth=300
        )
        gui.button(self.controlArea, self, "Read", callback=self.read)

    def browse_file(self):
        """Call back: enable file dialog and get the path of the select file."""
        start_dir = os.path.expanduser("./")
        self.fpath, _ = QFileDialog.getOpenFileName(self, "Open File", start_dir, "")

        # format the path label
        text = self.label_path.text()
        text_len_limit = 40
        if len(text) > text_len_limit:
            text = "..." + text[-text_len_limit:]
            self.label_path.setText(text)

    def read(self):
        """Call back: read file from json and output dataframe"""
        data = read_json_file(self.fpath)
        data = data.dropna().reset_index(drop=True)
        table = table_from_frame(data)
        self.Outputs.output_data.send(table)
