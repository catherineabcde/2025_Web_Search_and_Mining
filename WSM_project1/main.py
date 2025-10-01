"""
Task 1
Given 7874 English News:
1. Calculate the relevance of each news by the given query.
2. Return the top-k relevant news IDs.
"""
# 1. remove stopwords and indexing
import VectorSpace as vs
import os

env_path = "./EnglishNews/"
eng_txts = [f for f in os.listdir(env_path) if f.endswith(".txt")]

for txt in range(len(eng_txts)):

    f = open(env_path + eng_txts[txt], "r+")
    tokens = parser.tokenise(f)
    f = parser.removeStopWords(tokens)
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    f.close()

# 2. Transfer queries into vector
