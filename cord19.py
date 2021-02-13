import json
import math
import os
import re
import string
import time
import nltk
import numpy
import sys
import threading
import psutil
import random

from langdetect import detect

process = psutil.Process(os.getpid())
starting_mem = process.memory_info().rss
def printmem():
    mem = process.memory_info().rss - starting_mem
    print(f" bytes: {mem:,}")

def detect_english(doc):
    # Default langauge is English (most docs are)
    # All we care about are the 'en' ones
    language = 'en'
    try:
        if len(doc) > 25:
            language = detect(' '.join(doc[:25]))
        elif len(doc) > 0:
            language = detect(' '.join(doc[:len(doc)]))
    except Exception as e:
        # If cannot detect language, drop doc
        return False
    return language == 'en'

def inverse_document_frequency(term, corpus):
    num_occurring = 0
    for doc in corpus:
        if term in doc:
            num_occurring += 1
    return math.log2(len(corpus)/num_occurring)

def tfidf(word_id, vocabulary, doc, corpus_length):
    idf = math.log2(corpus_length/vocabulary['words'][word_id][1])
    term_frequency = doc[1][word_id] / doc[2]
    return term_frequency * idf

timings = {
    'load': 0,
    'lowercase': 0,
    'punctuation': 0,
    'tokenize': 0,
    'stopwords': 0,
    'stemming': 0,
    'vocab': 0,
    'openfile': 0,
    'detect_english': 0,
    'is_number': 0,
}

stopwords = set(nltk.corpus.stopwords.words('english'))

def prep_doc(filepath, vocab):
    t1 = time.time()
    with open(filepath) as file:
        timings['openfile'] += time.time()-t1
        t1 = time.time()
        doc = json.load(file)
        timings['load'] += time.time()-t1
        # get text and lowercase it, then combine
        t1 = time.time()
        text = ' '.join([chunk['text'].lower() for chunk in doc['body_text']])
        timings['lowercase'] += time.time()-t1
        # remove punctuation
        t1 = time.time()
        regex = re.compile(f"[{re.escape(string.punctuation)}]")
        text = regex.sub('', text)
        timings['punctuation'] += time.time()-t1
        # tokenize, remove empty words, get a sample
        t1 = time.time()
        text = text.split(' ')
        text = [word for word in text if word != '']
        timings['tokenize'] += time.time()-t1
        # Check langauge - if not English, do not include it
        t1 = time.time()
        if not detect_english(text):
            return
        timings['detect_english'] += time.time()-t1
        # remove stopwords
        t1 = time.time()
        text = [word for word in text if word not in stopwords]
        timings['stopwords'] += time.time()-t1
        # remove numbers
        t1 = time.time()
        regex = re.compile(f"^[0-9-−]+$")
        text = [word for word in text if not regex.match(word)]
        timings['is_number'] += time.time()-t1
        # Lemmatization
        t1 = time.time()
        lemmatizer = nltk.wordnet.WordNetLemmatizer()
        text = [lemmatizer.lemmatize(word) for word in text]
        timings['stemming'] += time.time()-t1

        t1 = time.time()
        doc_words = {}
        for word in text:
            word_id = None
            # add or find in vocab
            if word in vocabulary['index']:
                word_id = vocabulary['index'][word]
            else:
                # new entry
                word_id = len(vocabulary['index'])
                vocabulary['index'][word] = word_id
                vocabulary['words'][word_id] = [word, 0]
            # create or increment word in this doc's word list
            # later the count will be replaced by tfidf
            if word_id in doc_words:
                doc_words[word_id] += 1
            else:
                doc_words[word_id] = 1
        # update corpus frequencies - each unique word in this doc adds 1
        for word in set(text):
            word_id = vocabulary['index'][word]
            vocabulary['words'][word_id][1] = vocabulary['words'][word_id][1] + 1
        timings['vocab'] += time.time()-t1
        # max_freq is being stored so we don't have to recalculate for the tfidf of every word
        try:
            max_freq = max(doc_words.values())
        except:
            print(f"Problem with doc: {filepath}")
            return None
        return [os.path.basename(filepath), doc_words, max_freq]

def get_n_docs(n, vocabulary):
    dirpath = '../input/CORD-19-research-challenge/document_parses/pdf_json'
    corpus = []

    i = 0
    stats = {
        'last_i': 0,
        'last_time': 0,
    }

    run_status_checker = True
    def status_checker():
        while run_status_checker:
            completed_per_sec = (i - stats['last_i']) / (time.time() - stats['last_time'])
            stats['last_i'] = i
            stats['last_time'] = time.time()
            print(f"{i} / {n} ({round(completed_per_sec, 1)}/sec)")
            printmem()
            time.sleep(1)

    threading.Thread(target=status_checker, daemon=True).start()

    for path in os.listdir(dirpath):
        i += 1
        if i > n:
            break
        corpus.append(prep_doc(os.path.join(dirpath, path), vocabulary))

    run_status_checker = False

    # Drop all NoneType docs (occurs if not English)
    corpus = [x for x in corpus if x != None]
    return corpus, vocabulary

def doc_tfidf(doc, vocabulary, corpus_length):
    doc_words = doc[1]
    for word_id in doc_words:
        doc_words[word_id] = tfidf(word_id, vocabulary, doc, corpus_length)
    return doc

def peek_vocab(vocabulary):
    print('--- First 5 vocabulary entries ---')
    i = 0
    for word_id in vocabulary['words']:
        i += 1
        if i > 5:
            break
        print(vocabulary['words'][word_id])

    print('--- Last 5 vocabulary entries ---')
    i = 0
    for word_id in vocabulary['words']:
        i += 1
        if i < len(vocabulary['words'])-4:
            continue
        print(vocabulary['words'][word_id])

def peek_doc(doc):
    print('--- Peeking at document ---')
    print(f"filename: {doc[0]}")
    print('--- First 5 unique words in doc ---')
    i = 0
    for word_id in doc[1]:
        i += 1
        if i > 5:
            continue
        actualword = vocabulary['words'][word_id][0]
        print(f"'{actualword}', {doc[1][word_id]}")
    print('--- Last 5 unique words in doc ---')
    i = 0
    for word_id in doc[1]:
        i += 1
        if i < len(doc[1])-4:
            continue
        actualword = vocabulary['words'][word_id][0]
        print(f"'{actualword}', {doc[1][word_id]}")

def vectorize(doc, vocab_length):
    # Inflate doc to vocabulary length
    doc_vector = []
    for i in range(vocab_length):
        if i in doc[1].keys():
            # If doc contains word in vocab, save it's tfidf
            doc_vector.append(doc[1][i])
        else:
            # Otherwise add a 0 for that word
            doc_vector.append(0)
    # Return vector of vocabulary length
    return doc_vector

def get_vectors(n, dimension):
    # These are our hash functions
    vectors = []
    for i in range(n):
        vector = []
        possibilities = [-1, 1]
        # Each sketch is a vector of only +1's and -1's
        for j in range(dimension):
            vector.append(possibilities[random.randint(0,1)])
        vectors.append(vector)
    return vectors

def signature(doc, vectors, a, vocab_length):
    def project_on_(u,v):
        # This is the scaler projection
        # Returns the amount (length) of u that is in the direction of v
        return numpy.dot(u,v) / numpy.linalg.norm(v)
    doc_vector = vectorize(doc, vocab_length)
    signature = []
    for vector in vectors:
        # Get the amount of doc_vector in the direction of sketch
        projection = project_on_(doc_vector, vector)
        # Find which bin the projection lands in based on paramter 'a'
        hash_bin = math.ceil(projection / a)
        signature.append(hash_bin)
    return signature

def corpus_signatures_LSH(corpus, sketches, a, r, vocab_length):
    # Combines signature generation and LSH steps
    def LSH(doc_sig, r):
        bands_hashes = []
        for i in range(b):
            band = doc_sig[i:i+r]
            # Use the built-in Python hash function to hash each band
            # We can change this as needed
            bands_hashes.append(hash(frozenset(band)))
        return bands_hashes
    corpus_new = []
    for i in range(len(corpus)):
        doc = corpus[i]
        doc_id = doc[0]
        # Get doc signature using vector projections
        doc_sig = signature(doc, sketches, a, vocab_length)
        # Then do LSH on signatures with b bands to get hashes
        doc_LSH = LSH(doc_sig, r)
        corpus_new.append((doc_id, doc_LSH))
    return corpus_new

def find_candidates(corpus, LSH_length):
    candidates = {}
    # Look at all pairs of docs
    for i in range(len(corpus)):
        doc1 = corpus[i]
        doc1_matches = []
        for j in range(len(corpus)):
            if j != i:
                doc2 = corpus[j]
                # Find if any have same hash value for any band
                for k in range(LSH_length):
                    if doc1[1][k] == doc2[1][k]:
                        doc1_matches.append(doc2[0])
        candidates[doc1[0]] = doc1_matches
    return candidates

def group_candidates(candidates):
    # TODO: FIX
    # Find groups (clusters) of documents that are all similar to each other
    grouped_candidates = {}
    group_id = 0
    # Once we consider a doc, we do not want to consider it again
    # A document can only belong to one group (cluster)
    exclusion_list = []
    for doc1 in list(candidates.keys()):
        docs_similar_to_doc1 = candidates[doc1]
        to_check = [x for x in docs_similar_to_doc1 if x not in exclusion_list]
        doc1_group = set([doc1])
        # Check if each doc that is similar to doc1 is also similar to other docs in to_check
        for i in range(len(to_check)):
            for j in range(i, len(to_check)):
                # If these document appear in each other's similarity lists, add them to group
                if to_check[j] in candidates[to_check[i]]:
                   doc1_group.add(to_check[i])
                   doc1_group.add(to_check[j])
        grouped_candidates[group_id] = list(doc1_group)
        exclusion_list.append(doc1)
        group_id += 1
    return grouped_candidates

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
peek_doc(doc)

print('--- Performance ---')
for timing in timings:
    print(f"{timing}: {timings[timing]}")

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
corpus_LSH_results = corpus_signatures_LSH(corpus, vectors, a, r, len(vocabulary['index']))
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
    group_id : [doc_id1, doc_id2,...]
}
'''

# TODO: write method to get top 5 words in each group (maybe just top 5 in first document of group)
