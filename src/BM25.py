from pyserini.search.lucene import LuceneSearcher
import os
import json


BM_searcher = LuceneSearcher.from_prebuilt_index('wikipedia-dpr-100w')

def BM_find_document(document: str, time = 5):
  hits = BM_searcher.search(document, time)
  paragraph = []
  for i in range(len(hits)):
    doc = BM_searcher.doc(hits[i].docid)
    json_doc = json.loads(doc.raw())
    paragraph.append(json_doc['contents'])
  return paragraph

def dspy_BM(q:str, time=10):
  hits = BM_searcher.search(q,k=time)
  run = {}
  for hit in hits:
    doc = BM_searcher.doc(hit.docid)
    json_doc = json.loads(doc.raw())
    run[json_doc['contents']] = float(hit.score)
  return run