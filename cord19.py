import time

from library import (corpus_signatures_LSH, doc_tfidf, find_candidates,
                     get_n_docs, get_vectors, group_candidates, peek_doc,
                     peek_vocab, printmem)

# vocabulary: {
#     'index': {
#         word: word_id
#     },
#     'words': {
#         word_id: (word, corpus_freq)
#     }
# }

# doc: [
#     doc_file_name,
#     {
#         word_id: doc_freq,
#     },
#     max_freq,
# ]

vocabulary = {
    'index': {}, # 'actualword': word_id
    'words': {}, # word_id: ['actualword', corpus_frequency]
}

printmem()

num_docs = 2
t0 = time.time() # Track total time

t1 = time.time()
corpus, vocabulary = get_n_docs(num_docs, vocabulary)
t2 = time.time()
print(f"prepping docs: {t2-t1}")

print(f"Number of words in vocabulary: {len(vocabulary['words'])}")

printmem()

t1 = time.time()
corpus_length = len(corpus)
corpus = [doc_tfidf(doc, vocabulary, corpus_length) for doc in corpus]
t2 = time.time()
print(f"tfidf: {t2-t1}")

printmem()

print(f"Corpus size: {corpus_length}")

t2 = time.time()
print(f'Total time: {t2-t0}')

doc = corpus[0]
peek_vocab(vocabulary)
peek_doc(doc, vocabulary)

###########

# Parameters for Signature Generation and LSH
a = 1 # This is the bucket size for signature generation
b = 2 # Number of bands: increase -> increase likelihood of finding documents are similar
r = 6 # Number of rows per band: increase -> decrease likelihood of finding documents are similar
# b * r = length of transformed docs = number of random vectors

t1 = time.time()
vectors = get_vectors(b*r, len(vocabulary['index']))
t2 = time.time()
print(f"random vectors: {t2-t1}")

t1 = time.time()
corpus_LSH_results = corpus_signatures_LSH(corpus, vectors, a, r, b, len(vocabulary['index']))
t2 = time.time()
print(f"signatures + LSH: {t2-t1}")
      
# corpus_LSH_results: [
#     doc_file_name,
#     [
#         hash_value_of_band,...
#     ]
# ]

'''
candidates: 
{
    doc_id1: 
    [
        all similar docs to doc1
    ]
}
'''

t1 = time.time()
grouped_candidates = group_candidates(find_candidates(corpus_LSH_results, b))
t2 = time.time()
print(f"candidates + grouping: {t2-t1}")

'''
grouped_candidates: 
{
    group_id : [doc1_corpus_index, doc2_corpus_index,...]
}
'''

t1 = time.time()
groups = top_group_words(grouped_candidates, corpus, vocabulary)
t2 = time.time()
print(f"finding group words: {t2-t1}")
'''
groups: 
[
    (
        [doc1_corpus_index, doc2_corpus_index,...],
        [largest_tfidf_word_doc1, largest_tfidf_word_doc2,...]
    )
]
'''
print('\nGroups and top words:')
print(groups)
