"""Argument topic modeling module.
"""

from typing import List
import pandas as pd
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import PartOfSpeech
import spacy
import copy
import torch
import networkx as nx
import numpy as np
from textblob import TextBlob

class ArgumentChunker:
    def __init__(self, docs:List[str]):
        self.docs = docs
        self.df_chunks = None
        self.topic_model = ArgumentTopic()
        
    def chunk(self):
        """Split argument into chunks.
        """
        if self.df_chunks and "chunk" in self.df_chunks.columns:
            return
        
        nlp_pip = "en_core_web_md"
        if nlp_pip not in spacy.cli.info()['pipelines']:
            spacy.cli.download(nlp_pip)
        nlp = spacy.load(nlp_pip)
        
        argt_ids = []
        chunks = []
        for i, doc in enumerate(self.docs):
            seen = set()
            doc = nlp(doc)
            for sent in doc.sents:
                heads = [c for c in sent.root.children if c.dep_ in ['conj']]
                for head in heads:
                    head_chunk = [w for w in head.subtree]
                    [seen.add(w) for w in head_chunk]
                    chunk = ' '.join([w.text for w in head_chunk]) 
                    argt_ids.append(i)
                    chunks.append(chunk)
                unseen = [w for w in sent if w not in seen]
                chunk = ' '.join([w.text for w in unseen])
                argt_ids.append(i)
                chunks.append(chunk)
        
        self.df_chunks = pd.DataFrame({'argument_id': argt_ids, 'chunk': chunks})
    
    def chunk_polarity_score(self):
        """Compute chunk polarity scores.
        """
        if "polarity_score" in self.df_chunks.columns:
            return
        
        self.df_chunks["polarity_score"] = self.df_chunks["chunk"].apply(
            lambda c: TextBlob(c).sentiment.polarity
        )
        
    def chunk_topic(self):
        """Compute chunk topics and probabilities.
        """
        if "topic" in self.df_chunks.columns:
            return 
        
        topics, probs = self.topic_model.fit_transform_reduced(
            self.df_chunks["chunk"])
        self.df_chunks["topic"] = topics
        self.df_chunks["topic_prob"] = probs
        
    def chunk_rank(self):
        """Compute sentence rank of chunks.
        """
        if "rank" in self.df_chunks.columns:
            return
        assert "topic" in self.df_chunks.columns, \
            "Should do topic modeling before computing chunk ranks!"
        
        ranks = self.topic_model.get_doc_rank()
        self.df_chunks["rank"] = ranks
        
    def get_chunk_table(self):
        """Get full info table of chunks that include topic, sentence rank, 
        and polarity
        """ 
        self.chunk()
        self.chunk_topic()
        self.chunk_rank()
        self.chunk_polarity_score()
        return copy.deepcopy(self.df_chunks)
    

# def merge_chunks(docs:List[str], doc_ids:List[int], topics:List[int], \
#     chunk_ranks:list[float], chunk_polarity_scores:List[float], \
#         topic_keywords:dict, n_keywords:int):
#     """For each argument, merge topics of chunks into one, in format of keyword and importance.

#     Args:
#         docs (List[str]): argument doc list.
#         doc_ids (list[int]): chunk's doc ids.
#         topics (list[int]): chunk's topic ids.
#         chunk_ranks: list[float]: chunkrank in chunk corpus.
#         chunk_polarity_scores: list[float]: chunk polarity scores.
#         topic_keywords (dict): all topic keywords and importance.
#     """
#     docs = pd.DataFrame({"doc": docs}) 
#     chunks = pd.DataFrame({
#         "doc_id": doc_ids, 
#         "topic": topics
#     })
    
#     def get_keywords_scores(topics:List[int]):
#         keywords = [topic_keywords[t] for t in topics if t != -1]
#         keywords = list(itertools.chain.from_iterable(keywords))
#         keywords = pd.DataFrame({
#             "keyword": [kw[0] for kw in keywords], 
#             "keyword_scores": [kw[1] for kw in keywords]
#         }) 
#         keywords = keywords.groupby("keyword", as_index=False).\
#             sum().sort_values(by="keyword_scores", ascending=False).reset_index(drop=True)
#         keywords = keywords.loc[0:n_keywords-1]
#         return keywords["keyword"].tolist(), keywords["keyword_scores"].tolist()
    
#     chunks["topic"] = chunks["topic"].apply(lambda x: [x])
#     docs["topic"] = chunks.groupby("doc_id", as_index=False).agg("sum")["topic"]
#     temp = docs["topic"].apply(get_keywords_scores)
#     docs[["keyword", "keyword_scores"]] = pd.DataFrame(temp.tolist(), index=temp.index)
    
#     return docs

class ArgumentTopic(BERTopic):
    
    def __init__(self):
        self.embedding_model = SentenceTransformer("all-mpnet-base-v1")
        self.umap_model = UMAP(n_components=5)
        self.hdbscan_model = HDBSCAN(min_cluster_size=10, prediction_data=True)
        self.vectorizer_model = CountVectorizer(stop_words='english', ngram_range=[1, 1])
        self.ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        
        # NOTE: The POS patterns should be discussed further.
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
            calculate_probabilities=False
        )
     
    def fit_transform_reduced(self, docs:List[str]):
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
    
    def get_topic_table(self):
        """Return topic information as a table.
        """
        topic_list = self.get_topics()
        topic_info = copy.deepcopy(self.get_topic_info()) 
        keywords = []
        keyword_scores = []
        
        for _, topic in topic_list.items():
            keywords.append([kw[0] for kw in topic])
            keyword_scores.append([kw[1] for kw in topic])
        topic_info["keyword"] = keywords
        topic_info["keyword_scores"] = keyword_scores
        topic_info.rename(columns={
            "Topic": "topic", 
            "Count": "count", 
            "Name": "name"
        })
        
        return topic_info
    
    def get_doc_rank(self):
        """compute sentence rank of docs in corpus.
        """
        embeds = self.umap_model.embedding_
        embeds = torch.tensor(embeds)
        embeds /= embeds.norm(dim=-1).unsqueeze(-1)
        sim_mat = embeds @ embeds.t()
        sim_mat = sim_mat.numpy(force=True)
        G = nx.from_numpy_array(sim_mat)
        ranks = list(nx.pagerank(G).values())
        ranks = np.array(ranks)
        ranks = ranks / ranks.max() # max normalization
        return ranks