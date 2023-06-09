"""Argument processor module.

This compute the usefulness of arguments that will be used for
pre-filtering the arguments for the following mining steps.
"""

import spacy
from spacy.language import Language
from spacy_readability import Readability
from importlib.util import find_spec

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
  
    def __init__(self, df_arguments):
        pipe_name = "en_core_web_md"
        if find_spec(pipe_name) is None:
            spacy.cli.download(pipe_name)
            
        self.df_arguments = df_arguments
        self.nlp_pipe = spacy.load(pipe_name)
        self.nlp_pipe.add_pipe('readability', last=True)
            
    def argument_readability(self):
        """Compute argument readability.
        """
        if 'readability' in self.df.columns:
            return
       
        docs = self.df_arguments['argument']
        docs = self.nlp_pipe.pipe(texts=docs)
        
        readabilities = [] 
        for doc in docs:
            readability = doc._.flesch_kincaid_reading_ease
            readabilities.append(readability)
        self.df_arguments['readability'] = readabilities
        
    def argument_topics(self, df_chunks):
        """Compute argument topics.
        """
        topics = df_chunks.groupby(by="argument_id", as_index=False).agg({
            "topic": lambda x: list(x)
        })["topic"]
        self.df_arguments["topics"] = topics
    
    def argument_sentiment(self, df_chunks):
        """Compute argument sentiment.
        """
        pass
    
    def argument_coherence(self, df_chunks):
        """Compute argument coherence.
        """
        pass
    
    def get_argument_table(self, df_chunks):
        """Get the processed argument table.
        """
        self.argument_readability()
        self.argument_topics(df_chunks)
        self.argument_coherence(df_chunks)
        return copy.deepcopy(self.df_arguments)