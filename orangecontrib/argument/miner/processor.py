"""Argument processor module.

This compute the usefulness of arguments that will be used for
pre-filtering the arguments for the following mining steps.
"""

import spacy
from spacy.language import Language
from spacy_readability import Readability
from flair.nn import Classifier
from flair.data import Sentence
from importlib.util import find_spec

# TODO: check if it's useful to replace this readability pipe with 
# py-readability-metrices (https://pypi.org/project/py-readability-metrics/)
@Language.component("readability")
def readability(doc):
    read = Readability()
    doc = read(doc)
    return doc


class ArgumentProcessor:
    """Process argument before mining.
    
    - Rename argument and score columns; 
    - Label arguments by usefulness.
    """
  
    def __init__(self, df, lang: str = "en"):
        if df is None:
            raise TypeError("Input data frame must not be None!")
        
        lang2name = {"en": "en_core_web_md"}
        pipe_name = lang2name[lang]
        if find_spec(pipe_name) is None:
            spacy.cli.download(pipe_name)
            
        self.df = df
        self.nlp_pipe = spacy.load(pipe_name)
        self.nlp_pipe.add_pipe('readability', last=True)
        self.sa_pipe = Classifier.load('sentiment')
            
    def rename_column(self, old_name, new_name):
        """Rename a column with old_name to new_name.
        """ 
        self.df.rename({old_name: new_name}, axis=1, inplace=True)
   
    def compute_readability(self, readable_theta: int = 4):
        """Compute readability of each argument.
        """
        if 'readability' in self.df.columns:
            return
       
        docs = self.df['argument'].astype('str')
        docs = self.nlp_pipe.pipe(texts=docs)
        
        readabilities = [] 
        for doc in docs:
            readability = doc._.flesch_kincaid_reading_ease
            readabilities.append(readability)
        v_min = min(readabilities)
        v_max = max(readabilities)    
        readabilities = [(r - v_min) * 10 / (v_max - v_min) for r in readabilities] 
        self.df['readability'] = readabilities
        self.df['readable'] = [r >= readable_theta for r in readabilities]
        
    def compute_sentiment(self):
        """Compute sentiment of each argument.
        """
        if 'sentiment' in self.df.columns:
            return 
        
        sentiments = [] 
        for index, row in self.df.iterrows():
            text = row['argument'] 
            sentence = Sentence(text)
            self.sa_pipe.predict(sentence)
            sentence = str(sentence)
            if 'POSITIVE' in sentence:
                sentiments.append(1)
            elif 'NEGATIVE' in sentence:
                sentiments.append(-1)
            else:
                sentiments.append(0)
        self.df['sentiment'] = sentiments
    
    def compute_coherence(self):
        """Compute coherence between the argument sentiment and its score
        """
        if 'coherence' in self.df.columns:
            return
        
        # this is based on the way how the flair sentiment model works    
        score2sentiment = {
            1: -1, 
            2: -1,
            3: -1, 
            4: 1, 
            5: 1
        }
        
        coherence = []
        for index, row in self.df.iterrows():
            c = row['sentiment'] == score2sentiment[row['score']]
            coherence.append(c)
        self.df['coherence'] = coherence
    
    def compute_usefulness(self):
        """Compute usefulness (0~2) of each argument. 
        """
        if 'usefulness' in self.df.columns:
            return
        
        usefulness = []  
        for index, row in self.df.iterrows():
            u = row['readable'] + row['coherence']
            usefulness.append(u)
        self.df['usefulness'] = usefulness
        
    def get_results(self):
        """return a copy of the argument data.
        """
        return self.df.copy()