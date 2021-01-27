import json
import math
import os
import re
import string
import time
import nltk
import numpy

from langdetect import detect

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

def term_frequency(term, doc):
    return doc[term] / max(doc.values())

def inverse_document_frequency(term, corpus):
    num_occurring = 0
    for doc in corpus:
        if term in doc:
            num_occurring += 1
    return math.log2(len(corpus)/num_occurring)

def tfidf(term, corpus, doc, word_document_frequencies):
    idf = math.log2(len(corpus)/word_document_frequencies[term])
    return term_frequency(term, doc) * idf

timings = {
    'load': 0,
    'lowercase': 0,
    'punctuation': 0,
    'tokenize': 0,
    'stopwords': 0,
    'stemming': 0,
}

def prep_doc(filepath):
    with open(filepath) as file:
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
        if not detect_english(text):
            return
        # remove stopwords
        t1 = time.time()
        stopwords = set(nltk.corpus.stopwords.words('english'))
        text = [word for word in text if word not in stopwords]
        timings['stopwords'] += time.time()-t1
        # Lemmatization
        t1 = time.time()
        lemmatizer = nltk.wordnet.WordNetLemmatizer()
        text = [lemmatizer.lemmatize(word) for word in text]
        timings['stemming'] += time.time()-t1
        docindex = dict.fromkeys(set(text))
        for word in docindex:
            docindex[word] = 0
        for word in text:
            docindex[word] += 1
        return docindex

def gen_corpus_frequencies_vocab(corpus):
    t1 = time.time()
    # get all words in corpus
    word_sets = [set(doc) for doc in corpus]
    corpus_words = set()
    for doc in word_sets:
        corpus_words.update(doc)
    timings['collectwords'] = time.time()-t1
    # get number of docs in which each word occurs
    t1 = time.time()
    wordcounts = dict.fromkeys(corpus_words)
    for word in wordcounts:
        wordcounts[word] = 0
    for word_set in word_sets:
        for word in word_set:
            wordcounts[word] += 1
    timings['countwords'] = time.time()-t1
    return wordcounts, list(corpus_words)

def get_n_docs(n):
    dirpath = '../input/CORD-19-research-challenge/document_parses/pdf_json'
    corpus = []
    i = 0
    for path in os.listdir(dirpath):
        i += 1
        if i > n:
            break
        corpus.append(prep_doc(os.path.join(dirpath, path)))
        #filepaths.append(os.path.join(dirpath, path))
    # Drop all NoneType docs (occurs if not English)
    corpus = [x for x in corpus if x != None]
    return corpus

def to_tfidf_vectors(corpus, vocabulary, word_document_frequencies):
    numerical_corpus = []
    for doc in corpus:
        num_vector_doc = []
        for word in vocabulary:
            if word not in doc:
                num_vector_doc.append(0)
            else:
                num_vector_doc.append(tfidf(word, corpus, doc, word_document_frequencies))
        numerical_corpus.append(num_vector_doc)
    return numerical_corpus

num_docs = 10
t0 = time.time() # Track total time

t1 = time.time()
corpus = get_n_docs(num_docs)
t2 = time.time()
print(f"prepping docs: {t2-t1}")

t1 = time.time()
word_document_frequencies, vocabulary = gen_corpus_frequencies_vocab(corpus)
print(f"Number of words in vocabulary: {len(vocabulary)}")
t2 = time.time()
print(f"gen_corpus_frequencies: {t2-t1}")

num_vector_corpus = to_tfidf_vectors(corpus, vocabulary, word_document_frequencies)
t2 = time.time()
print(numpy.count_nonzero(num_vector_corpus[0]))
print(f'Total time: {t2-t0}')

'''
Might be better/faster to use this tfidf method
https://www.kaggle.com/maksimeren/covid-19-literature-clustering#PCA--&-Clustering

from sklearn.feature_extraction.text import TfidfVectorizer
def vectorize(text, maxx_features):
    
    vectorizer = TfidfVectorizer(max_features=maxx_features)
    X = vectorizer.fit_transform(text)
    return X
'''

from sklearn.decomposition import PCA
# Looks like shape is wrong - are documents rows or columns?
# Either way, documents should be vocabulary length, not

print(f'Shape prior to reduciton: {numpy.shape(num_vector_corpus)}')
t1 = time.time()
corpus_reduced = PCA(n_components=0.95, random_state=42).fit_transform(numpy.array(num_vector_corpus))
t2 = time.time()
print(f'Shape prior to reduciton: {numpy.shape(corpus_reduced)}')
print(f'PCA time: {t2-t1}')
