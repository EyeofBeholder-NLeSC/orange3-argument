from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget
from Orange.data.pandas_compat import table_from_frame, table_to_frame
from orangecontrib.argument.miner.miner import ArgumentProcessor

class OWArgProcessor(OWWidget):
    """Argument processor widget
    
    Functions:
    - Indicating the argument content and score columns
    - Filtering arguments by their usefulness
    """
    
    name = 'Argument Processor'
    description = 'Process input argument table and prepare it for the mining step.'
    icon = 'icons/OWArgProcessor.svg'
    
    want_main_area = False
    
    # GUI variables
    col_argument = Setting('')
    col_score = Setting('')
    
    class Inputs:
        input_data = Input('Data', Table)
        
    class Outputs:
        output_data = Output('Data', Table) 
       
    def __init__(self):
        super().__init__() 
        
        # GUI
        pass
    
    @Inputs.input_data
    def set_input_data(self, data):
        self.input_data = data
        
    def commit(self):
        pass