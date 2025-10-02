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
import util

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

eng_vec = vs.VectorSpace(eng_docs)

"""
how many documents are there? object.documentVectors
how many words are there in these documents? object.vectorKeywordIndex
"""

# (1) TF Weighting (Raw TF in course PPT) + Cosine Similarity
# retrieve the top 10 relevant docs
query = "planet Taiwan typhoon"
query_vec = eng_vec.buildQueryVector(query)

# Consine Similarity
similarity = {}
for i, doc_vec in enumerate(eng_vec.documentVectors):
    similarity[eng_filename[i]] = util.cosine(doc_vec, query_vec)

top_10 = sorted(similarity.items(), key=lambda x: x[1], reverse=True)[:10]

# formulate
print("TF Cosine")
print(f"{'NewsID':<20} {'Score':<15}")
for filename, score in top_10:
    print(f"{filename:<20} {score:<15f}")
print("-" * 35)

# (2) TF-IDF Weighting + Cosine Similarity
# retrieve the top 10 relevant docs



    

















