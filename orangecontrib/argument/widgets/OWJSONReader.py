from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Output, OWWidget
from Orange.data.pandas_compat import table_from_frame
from AnyQt.QtWidgets import QFileDialog
from AnyQt.QtCore import Qt
from orangecontrib.argument.miner.reader import read_json_file
import os
    

class OWJSONReader(OWWidget):
    """Json file reader widget
   
    This widget reads a local JSON file and output the data as table.
    """
    name = "JSON Reader"
    description = "Read a local JSON file and output the data as table."
    icon = "icons/OWJSONReader.svg"
    want_main_area = False
    resizing_enabled = False
    
    # GUI variables
    fpath = Setting('')
    
    class Outputs:
        output_data = Output('Data', Table)
    
    def __init__(self):
        super().__init__()
       
        # GUI
        box_select_file = gui.widgetBox(self.controlArea, orientation=Qt.Horizontal, 
                                      box="Select File")
        button_browse = gui.button(box_select_file, self, "...", callback=self.browse_file)
        self.label_path = gui.label(box_select_file, self, "Path: %(fpath)s", labelWidth=300)
        
        button_read = gui.button(self.controlArea, self, "Read", callback=self.read)

    def browse_file(self):
        """Call back: enable file dialog and get the path of the select file.
        """
        start_dir = os.path.expanduser('./')
        self.fpath, _ = QFileDialog.getOpenFileName(self, 'Open File', start_dir, "")
        
        # format the path label
        text = self.label_path.text() 
        text_len_limit = 40
        if len(text) > text_len_limit:
            text = "..." + text[-text_len_limit:]
            self.label_path.setText(text)
        
    def read(self):
        """Call back: read file from json and output dataframe
        """
        df = read_json_file(self.fpath) 
        table = table_from_frame(df)
        self.Outputs.output_data.send(table)

        
if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWJSONReader).run()