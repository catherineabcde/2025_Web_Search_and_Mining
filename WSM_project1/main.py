"""
Task 1
Given 7874 English News:
1. Calculate the relevance of each news by the given query.
2. Return the top-k relevant news IDs.
"""
# 1. remove stopwords and indexing
import VectorSpace as vs
import Parser
import os


eng_path = "./EnglishNews/"
eng_txts = [f for f in os.listdir(eng_path) if f.endswith(".txt")]
eng_docs = []

for txt in eng_txts:
    file_path = os.path.join(eng_path, txt)
    with open(file_path, "r+") as f:
        text = f.read()
        eng_docs.append(text)
    f.close()

eng_vs = vs.VectorSpace(eng_docs)
print("how many documents are there?", len(eng_vs.documentVectors))
print("how many words are there?", len(eng_vs.vectorKeywordIndex))









# 2. Transfer queries into vector






