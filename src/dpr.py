import os
import json
from pyserini.search.faiss import FaissSearcher, DkrrDprQueryEncoder, DprQueryEncoder, AutoQueryEncoder,AggretrieverQueryEncoder
from pyserini.encode import TctColBertQueryEncoder
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.hybrid import HybridSearcher

sparse_searcher = LuceneSearcher.from_prebuilt_index('wikipedia-dpr-100w')


encoder = DprQueryEncoder("facebook/dpr-question_encoder-multiset-base")
# Initialize the searcher with the prebuilt FAISS index and encoder

dpr_searcher = FaissSearcher.from_prebuilt_index(
    #'wikipedia-dpr-100w.dkrr-nq',
    'wikipedia-dpr-100w.dpr-multi',
    encoder
)
hybrid_searcher = HybridSearcher(dpr_searcher, sparse_searcher)
def dpr_find_document(document: str,t=10):
  hits = dpr_searcher.search(document, t)
  paragraph = []
  for i in range(len(hits)):
    doc = dpr_searcher.doc(hits[i].docid)
    json_doc = json.loads(doc.raw())
    paragraph.append(json_doc['contents'])
    
    #paragraph += "\n" + doc.raw()
  return paragraph

def hybrid_find_document(document: str,t=10):
  hits = hybrid_searcher.search(document, t)
  paragraph = []
  for i in range(len(hits)):
    doc = dpr_searcher.doc(hits[i].docid)
    json_doc = json.loads(doc.raw())
    paragraph.append(json_doc['contents'])

    #paragraph += "\n" + doc.raw()
  return paragraph