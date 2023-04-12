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
    usefulness_theta = Setting(0)
    
    class Inputs:
        input_data = Input('Data', Table)
        
    class Outputs:
        output_data = Output('Data', Table) 
       
    def __init__(self):
        super().__init__() 
        
        # GUI
        rename_columns = gui.vBox(self.controlArea, box='Rename')
        self.rename_arg = gui.comboBox(rename_columns, self, 'col_argument', 
                                       label='Column of Argument Text', 
                                       sendSelectedValue=True, 
                                       items=())
        self.rename_score = gui.comboBox(rename_columns, self, 'col_score', 
                                         label='Column of Argument Score', 
                                         sendSelectedValue=True, 
                                         items=())
        filter = gui.vBox(self.controlArea, box='Filter')
        self.usefulness_filter = gui.hSlider(filter, self, 'usefulness_theta', 
                                             minValue=0, maxValue=2)
        gui.button(self.controlArea, self, 
                   label='Process', 
                   callback=self.process)
    
    @Inputs.input_data
    def set_input_data(self, data):
        self.input_data = data
        df = table_to_frame(data, include_metas=True)
        items = list(df.columns)
        
        # udpate combobox
        self.rename_arg.clear()
        self.rename_score.clear()
        self.rename_arg.addItems(items)
        self.rename_score.addItems(items)
        self.col_argument = items[0]
        self.col_score = items[0]
        
    def process(self):
        processor = ArgumentProcessor(
            table_to_frame(self.input_data, include_metas=True)
        )
        
        if not {'argument', 'score'}.issubset(processor.df.columns): 
            processor.rename_column(self.col_argument, 'argument')
            processor.rename_column(self.col_score, 'score')
        processor.compute_textrank()
        processor.compute_readability()
        processor.compute_sentiment()
        processor.compute_coherence()
        processor.compute_usefulness()
        result = processor.filter(usefulness_theta=self.usefulness_theta)
        
        self.Outputs.output_data.send(table_from_frame(result))
        