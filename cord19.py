import math
import json
import string
import nltk
import os
import time

def sorted_frequencies(doc):
    frequencies = []
    for word in doc:
        entry = next((e for e in frequencies if e['word'] == word), None)
        if entry is None:
            frequencies.append({'word': word, 'count': 1})
        else:
            entry['count'] += 1
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

def prep_doc(filepath):
    with open(filepath) as file:
        doc = json.load(file)
        # get text and lowercase it, then combine
        text = ' '.join([chunk['text'].lower() for chunk in doc['body_text']])
        # remove punctuation
        text = ''.join([char for char in text if char not in string.punctuation])
        # tokenize, remove empty words, get a sample
        text = text.split(' ')
        text = [word for word in text if word != '']
        # remove stopwords
        stopwords = nltk.corpus.stopwords.words('english')
        text = [word for word in text if word not in stopwords]
        # stemming
        stemmer = nltk.stem.LancasterStemmer()
        text = [stemmer.stem(word) for word in text]
        return text

def gen_corpus_frequencies(corpus):
    # get all words in corpus
    corpus_words = set()
    for doc in corpus:
        corpus_words.update(doc)
    # get number of docs in which each word occurs
    wordcounts = dict.fromkeys(corpus_words)
    for word in wordcounts:
        wordcounts[word] = 0
        for doc in corpus:
            if word in doc:
                wordcounts[word] += 1
    return wordcounts

limit = 100
i = 0
filepaths = []
dirpath = 'data/document_parses/pdf_json'
for path in os.listdir(dirpath):
    i += 1
    if i > limit:
        break
    filepaths.append(os.path.join(dirpath, path))

print(filepaths)

t1 = time.time()
docs = [prep_doc(path) for path in filepaths]
t2 = time.time()
print(f"prepping docs: {t2-t1}")

t1 = time.time()
word_document_frequencies = gen_corpus_frequencies(docs)
print(len(word_document_frequencies))
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
