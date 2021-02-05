import numpy as np
from random import uniform, randint
import math

'''
At this point have corpus in form
[(doc1_id, {first occuring word in document's vocab index: tfidf, second occuring word... : tfidf}, ...), (doc2_id, {...]
'''

def vectorize(doc):
    # TODO: add vector inflation here:
    # return vector of vocabulary length, not just doc length
    return list(doc[1].values())

def n_sketches(n, dimension):
    # These are our hash functions
    # Vectors we will project our document tfidf vectors onto
    sketches = []
    for i in range(n):
        sketch = []
        possibilities = [-1, 1]
        # Each sketch is a vector of only +1's and -1's
        # TODO: may need to change this since all our doc vector values will be non-negative
        for j in range(dimension):
            sketch.append(possibilities[randint(0,1)])
        sketches.append(sketch)
    return sketches

def project_on_(u,v):
    # This is the scaler projection
    # Returns the amount (length) of u that is in the direction of v
    return np.dot(u,v) / np.linalg.norm(v)

def signature(doc, sketches, a):
    doc_vector = vectorize(doc)
    signature = []
    for sketch in sketches:
        projection = project_on_(doc_vector,sketch)
        hash_bin = math.ceil(projection / a)
        signature.append(hash_bin)
    return signature

def corpus_signatures(corpus, sketches, a):
    # Changes each document into (id, [signature])
    for i in range(len(corpus)):
        doc = corpus[i]
        doc_sig = (doc[0], signature(doc,sketches,a))
        corpus[i] = doc_sig
    return corpus

'''
Running everything
'''

vocab_size = 5
num_docs = 2
num_hashes = 3
a = 1 # This is a parameter for the buckets of the hash functions

sketches = n_sketches(num_hashes, vocab_size)
corpus = corpus_signatures(corpus, sketches, a) # already have corpus as described above

'''
Next is to use bands technique to hash signatures
'''
