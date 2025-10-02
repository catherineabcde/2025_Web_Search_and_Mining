"""
Task 1
Given 7874 English News:
1. Calculate the relevance of each news by the given query.
2. Return the top-k relevant news IDs.
"""
# 1. remove stopwords and indexing 2. Transfer documents into vector
import VectorSpace as vs
import Parser
import os
import tfidf

eng_path = "./EnglishNews/"
eng_txts = [f for f in os.listdir(eng_path) if f.endswith(".txt")]
eng_docs = []
eng_filename = []

for txt in eng_txts:
    file_path = os.path.join(eng_path, txt)
    with open(file_path, "r+") as f:
        text = f.read()
        eng_docs.append(text)
        eng_filename.append(txt)
    f.close()

eng_vs = vs.VectorSpace(eng_docs)

"""
how many documents are there? object.documentVectors
how many words are there in these documents? object.vectorKeywordIndex
"""

# (1) TF Weighting (Raw TF in course PPT) + Cosine Similarity
# retrieve the top 10 relevant docs
query = "planet Taiwan typhoon"
tf_score = {}

















