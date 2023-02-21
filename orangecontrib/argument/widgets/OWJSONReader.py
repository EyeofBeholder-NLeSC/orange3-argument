from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import  Output, OWWidget
from Orange.data.pandas_compat import table_from_frame
from Orange.widgets.utils.filedialogs import open_filename_dialog
from AnyQt.QtWidgets import QGridLayout, QStyle, QFileDialog
from AnyQt.QtCore import Qt

import pandas as pd
import os


LOCAL_FILE, URL = (0, 1)
    

class OWJSONReader(OWWidget):
    """Json file reader widget
   
    This widget should allow reading file in JSON format from either local or remote (URL). 
    """
    
    name = "JSON Reader"
    description = "Read a JSON file from either local or remote URL."
    icon = "icons/OWJSONReader.svg"
    
    want_main_area = False
    
    # GUI variables
    source = Setting(LOCAL_FILE)
    fpath = Setting('')
    url = Setting('')
    file_loc = Setting('')
    
    class Outputs:
        output_data = Output('Data', Table)
    
    def __init__(self):
        super().__init__()
       
        # GUI
        layout = QGridLayout()
        layout.setSpacing(6)
        
        gui.widgetBox(self.controlArea, orientation=layout, box='Source')
        rbuttons = gui.radioButtons(None, self, 'source', box=True, 
                                callback=self.switch_control, addToLayout=False)
         
        file_radio = gui.appendRadioButton(rbuttons, 'File:', addToLayout=False)
        self.file_button = gui.button(None, self, '...', 
                                 callback=self.browse_file, autoDefault=False)
        self.file_button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.file_button.setEnabled(self.source == LOCAL_FILE)
        layout.addWidget(file_radio, 0, 0, Qt.AlignVCenter)
        layout.addWidget(self.file_button, 0, 1)
        
        url_radio = gui.appendRadioButton(rbuttons, 'URL:', addToLayout=False)
        self.url_edit = gui.lineEdit(None, self, 'url', 
                                     callback=self.edit_url, callbackOnType=self.edit_url)
        self.url_edit.setEnabled(self.source == URL)
        layout.addWidget(url_radio, 3, 0, Qt.AlignVCenter)
        layout.addWidget(self.url_edit, 3, 1)
        
        load_button = gui.button(None, self, 'Load', 
                                 callback=self.load_data, autoDefault=False)
        self.path_label = gui.label(None, self, '%(file_loc)s', 
                                    addToLayout=False, labelWidth=300)
        self.format_path_label()
        layout.addWidget(load_button, 5, 0)
        layout.addWidget(self.path_label, 5, 1)

    def switch_control(self):
        if self.source == LOCAL_FILE:
            self.file_button.setEnabled(True)
            self.url_edit.setEnabled(False)
            self.file_loc = self.fpath
        elif self.source == URL:
            self.file_button.setEnabled(False)
            self.url_edit.setEnabled(True)
            self.file_loc = self.url
        self.format_path_label()
            
    def browse_file(self):
        start_dir = os.path.expanduser('~/')
        self.fpath, _ = QFileDialog.getOpenFileName(self, 'Open File', start_dir, "")
        self.file_loc = self.fpath
        self.format_path_label()
        
    def edit_url(self):
        self.file_loc = self.url
        self.format_path_label()
        
    def format_path_label(self):
        len_limit = 40
        text = self.path_label.text()
        if len(text) <= len_limit:
            return
        else:
            text = '...' + text[-len_limit:]
            self.path_label.setText(text)
        
    def load_data(self):
        try:
            df = pd.read_json(self.file_loc)
        except ValueError:
            df = pd.read_json(self.file_loc, lines=True) 
        self.Outputs.output_data.send(table_from_frame(df))

        
if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview  # since Orange 3.20.0
    WidgetPreview(OWJSONReader).run()