import pytest
from chunker import ArgumentChunker, ArgumentTopic

class TestClassTopic:
    topic_model = ArgumentTopic()
    
    # NOTE: it seems that all the arguments are classified 
    # as outliers, so can't do reduction further. Need to
    # an larger test set, or use the full set. And when using
    # the full set, remember to transfer dtype to str.
    def test_fit_transform_reduced(self, input_df):
        docs = input_df["argument"]
        self.topic_model.fit_transform_reduced(docs)  