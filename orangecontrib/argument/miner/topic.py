"""Argument topic modeling module.
"""

from typing import List
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import PartOfSpeech
import spacy


def chunker(docs:List[str]):
    """Split argument docs into chunks by dependency parsing.
    """
    nlp_pip = "en_core_web_md"
    if nlp_pip not in spacy.cli.info()['pipelines']:
        spacy.cli.download(nlp_pip)
    nlp = spacy.load(nlp_pip)
    
    doc_ids = []
    chunks = []
    for i, doc in enumerate(docs):
        seen = set()
        doc = nlp(doc)
        for sent in doc.sents:
            heads = [c for c in sent.root.children if c.dep_ in ['conj']]
            for head in heads:
                head_chunk = [w for w in head.subtree]
                [seen.add(w) for w in head_chunk]
                chunk = ' '.join([w.text for w in head_chunk]) 
                doc_ids.append(i)
                chunks.append(chunk)
            unseen = [w for w in sent if w not in seen]
            chunk = ' '.join([w.text for w in unseen])
            doc_ids.append(i)
            chunks.append(chunk)
    
    return pd.DataFrame({'doc_id': doc_ids, 'chunk': chunks})    

class ArguTopic(BERTopic):
    
    def __init__(self):
        self.embedding_model = SentenceTransformer("all-mpnet-base-v1")
        self.umap_model = UMAP(n_components=5)
        self.hdbscan_model = HDBSCAN(min_cluster_size=10, prediction_data=True)
        self.vectorizer_model = CountVectorizer(stop_words='english', ngram_range=[1, 1])
        self.ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        self.representation_model = PartOfSpeech("en_core_web_md", pos_patterns=[
            [{"POS": "NOUN"}], 
            [{"POS": "ADJ"}], 
            [{"POS": "VERB"}]
        ])

        super().__init__(
            embedding_model=self.embedding_model, 
            umap_model=self.umap_model, 
            hdbscan_model=self.hdbscan_model, 
            vectorizer_model=self.vectorizer_model, 
            ctfidf_model=self.ctfidf_model, 
            representation_model=self.representation_model, 
            calculate_probabilities=True
        )
        
    def fit_transform_reduce_outliers(self, docs: List[str]):
        """Fit documents and reduce outliers by default.
        """
        topics, probs = self.fit_transform(docs)
        new_topics = self.reduce_outliers(docs, topics, strategy="c-tf-idf", threshold=0.1)
        new_topics = self.reduce_outliers(docs, new_topics, strategy="distributions")
        topics = new_topics
        
        self.update_topics(
            docs, topics=new_topics, 
            vectorizer_model=self.vectorizer_model, 
            ctfidf_model=self.ctfidf_model, 
            representation_model=self.representation_model
        )
        
        return topics, probs
    
    