import json
import math
import os
import re
import string
import time
import nltk

from langdetect import detect

def detect_english(doc):
    # Default langauge is English (most docs are)
    language = 'en'
    if len(doc) > 25:
        language = detect(' '.join(doc[:25]))
    elif len(doc) > 0:
        language = detect(' '.join(doc[:len(doc)]))
    # This is a 2 letter string indicating the language
    # All we care about are the 'en' ones
    return language == 'en'

def sorted_frequencies(doc):
    frequencies = []
    for word in doc:
        frequencies.append({'word': word, 'count': doc[word]})
    frequencies.sort(key=lambda entry: -entry['count'])
    return frequencies

def term_frequency(term, sorted_frequencies):
    entry = next((e for e in sorted_frequencies if e['word'] == term), None)
    term_freq = entry['count']
    max_freq = sorted_frequencies[0]['count']
    return term_freq / max_freq

def inverse_document_frequency(term, corpus):
    num_occurring = 0
    for doc in corpus:
        if term in doc:
            num_occurring += 1
    return math.log2(len(corpus)/num_occurring)

def tfidf(term, corpus, sorted_frequencies, word_document_frequencies):
    idf = math.log2(len(corpus)/word_document_frequencies[term])
    return term_frequency(term, sorted_frequencies) * idf

timings = {
    'load': 0,
    'lowercase': 0,
    'punctuation': 0,
    'tokenize': 0,
    'stopwords': 0,
    'stemming': 0,
    'docindex': 0,
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
        # stemming
        t1 = time.time()
        lemmatizer = nltk.wordnet.WordNetLemmatizer()
        text = [lemmatizer.lemmatize(word) for word in text]
        timings['stemming'] += time.time()-t1
        t1 = time.time()
        docindex = dict.fromkeys(set(text))
        for word in docindex:
            docindex[word] = 0
        for word in text:
            docindex[word] += 1
        timings['docindex'] += time.time()-t1
        return docindex

def gen_corpus_frequencies(corpus):
    # get all words in corpus
    t1 = time.time()
    word_sets = [set(doc.keys()) for doc in docs]
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
    return wordcounts

limit = 10
i = 0
filepaths = []
dirpath = 'data/document_parses/pdf_json'
for path in os.listdir(dirpath):
    i += 1
    if i > limit:
        break
    filepaths.append(os.path.join(dirpath, path))

t1 = time.time()
docs = [prep_doc(path) for path in filepaths]
# Drop all NoneType docs (occurs if not English)
docs = [x for x in docs if x != None]
t2 = time.time()
print(f"prepping docs: {t2-t1}")

t1 = time.time()
word_document_frequencies = gen_corpus_frequencies(docs)
print(f"Number of words: {len(word_document_frequencies)}")
t2 = time.time()
print(f"gen_corpus_frequencies: {t2-t1}")

doc = docs[0]

t1 = time.time()
doc_stats = sorted_frequencies(doc)
t2 = time.time()
print(f"sorted_frequencies: {t2-t1}")

t1 = time.time()
for entry in doc_stats:
    entry['tfidf'] = tfidf(entry['word'], docs, doc_stats, word_document_frequencies)
t2 = time.time()
print(f"tfidf: {t2-t1}")

doc_stats.sort(key=lambda entry: -entry['tfidf'])

for entry in doc_stats[0:10]:
    print(entry['word'], entry['tfidf'])

for t in timings:
    print(f"{t}: {timings[t]}")
