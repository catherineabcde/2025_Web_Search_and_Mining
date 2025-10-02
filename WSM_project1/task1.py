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
import util
import math

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
eng_processed_docs = eng_vec.documentVectors
eng_words = eng_vec.vectorKeywordIndex

# (1) TF Weighting (Raw TF in course PPT) + Cosine Similarity
# retrieve the top 10 relevant docs
query = "planet Taiwan typhoon"
query_vec = eng_vec.makeVector(query)

# Consine Similarity
similarity1 = {}
for i, doc_vec in enumerate(eng_processed_docs):
    similarity1[eng_filename[i]] = util.cosine(doc_vec, query_vec)

top_10 = sorted(similarity1.items(), key=lambda x: x[1], reverse=True)[:10]

# formulate
print("TF Cosine")
print(f"{'NewsID':<20} {'Score':<15}")
for filename, score in top_10:
    print(f"{filename:<20} {score:<15f}")
print("-" * 35)

# (2) TF-IDF Weighting + Cosine Similarity

def idf(word_index, all_doc_vecs):
    """
    idf = log(total number of documents / (1 + number of documents that contain the word))
    add 1 to denominator to avoid division by zero
    """
    # how many docs contain this word
    n_contain = sum(1 for doc_vec in all_doc_vecs if doc_vec[word_index] > 0)
    return math.log(len(all_doc_vecs) / (1 + n_contain))

# IDF score for each word
idf_score = {}
for word, index in eng_words.items():
    idf_score[word] = idf(index, eng_processed_docs)

eng_tf_vec = []
eng_tfidf_vec = []

# TF-IDF vector for each document
for doc_vec in eng_processed_docs:

    total_word_count = sum(doc_vec)
    doc_tf_vec = []
    doc_tfidf_vec = []
    for word, index in eng_words.items(): # (keyword, index of the word)
        # tf: Raw TF
        tf = doc_vec[index]
        # tf: normalized TF
        # tf = doc_vec[index] / total_word_count if total_word_count > 0 else 0
        # TF
        doc_tf_vec.append(tf)
        # TF-IDF
        doc_tfidf_vec.append(tf * idf_score[word])
    
    eng_tf_vec.append(doc_tf_vec)
    eng_tfidf_vec.append(doc_tfidf_vec)
    

# Turn query into TF-IDF vector
total_query_word_count = sum(query_vec)
query_tf_vec = []
query_tfidf_vec = []
for word, index in eng_words.items():
    
    # tf: Raw TF
    tf = query_vec[index]
    # tf: normalized TF
    # tf = query_vec[index] / total_query_word_count if total_query_word_count > 0 else 0
    # TF
    query_tf_vec.append(tf)
    # TF-IDF
    query_tfidf_vec.append(tf * idf_score[word])

# Cosine Similarity
similarity2 = {}
for i, doc_tfidf_vec in enumerate(eng_tfidf_vec):
    similarity2[eng_filename[i]] = util.cosine(doc_tfidf_vec, query_tfidf_vec)

top_10 = sorted(similarity2.items(), key=lambda x: x[1], reverse=True)[:10]

# formulate
print("TF-IDF Cosine")
print(f"{'NewsID':<20} {'Score':<15}")
for filename, score in top_10:
    print(f"{filename:<20} {score:<15f}")
print("-" * 35)


# (3) TF Weighting + Euclidean Similarity

def Euclidean_Distance(doc_vec, query_vec):
    Dsquare = 0
    for i in range(len(doc_vec)):
        Dsquare += math.pow((doc_vec[i] - query_vec[i]), 2)
    return math.sqrt(Dsquare)

def Euclidean_Similarity(doc_vec, query_vec):
    return 1 / (1 + Euclidean_Distance(doc_vec, query_vec))

similarity3 = {}
for i, doc_tf_vec in enumerate(eng_tf_vec):
    similarity3[eng_filename[i]] = Euclidean_Similarity(doc_tf_vec, query_tf_vec)

top_10 = sorted(similarity3.items(), key=lambda x: x[1], reverse=True)[:10]

# formulate
print("TF Euclidean Similarity")
print(f"{'NewsID':<20} {'Score':<15}")
for filename, score in top_10:
    print(f"{filename:<20} {score:<15f}")
print("-" * 35)


# (4) TF Weighting + Euclidean Distance
similarity4 = {}
for i, doc_tfidf_vec in enumerate(eng_tfidf_vec):
    similarity4[eng_filename[i]] = Euclidean_Distance(doc_tfidf_vec, query_tfidf_vec)

top_10 = sorted(similarity4.items(), key=lambda x: x[1], reverse=False)[:10]

# formulate
print("TF-IDF Euclidean Distance")
print(f"{'NewsID':<20} {'Score':<15}")
for filename, score in top_10:
    print(f"{filename:<20} {score:<15f}")
print("-" * 35)






















