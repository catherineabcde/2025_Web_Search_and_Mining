"""
Task 3
Given 2,589 Chinese News:
1. Calculate the relevance of each news by the given query.
2. Return the top-k relevant news IDs.
"""
# 1. remove stopwords and indexing 2. Transfer documents into vector
import VectorSpace as vs
import Parser
import os
import util
import math

ch_path = "./ChineseNews/"

def Turn2Vec(path):
    txts =  [f for f in os.listdir(path) if f.endswith(".txt")]
    docs = []
    filename = []

    for txt in txts:
        file_path = os.path.join(path, txt)
        with open(file_path, "r+") as f:
            text = f.read()
            docs.append(text)
            filename.append(txt)
    f.close()

    return vs.VectorSpace(docs)

ch_txts = [f for f in os.listdir(ch_path) if f.endswith(".txt")]
ch_docs = []
ch_filename = []

for txt in ch_txts:
    file_path = os.path.join(ch_path, txt)
    with open(file_path, "r+") as f:
        text = f.read()
        ch_docs.append(text)
        ch_filename.append(txt)
    f.close()

eng_vec = vs.VectorSpace(eng_docs)

# (1) TF Weighting + Cosine Similarity

# (2) TF-IDF Weighting + Cosine Similarity