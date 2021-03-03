import json
import math
import os
import re
import string
import time
import nltk
import numpy
import threading
import psutil
import random
import pathlib
import pickle
from matplotlib import pyplot
import seaborn
from sklearn.decomposition import PCA

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
            if word in vocab['index']:
                word_id = vocab['index'][word]
            else:
                # new entry
                word_id = len(vocab['index'])
                vocab['index'][word] = word_id
                vocab['words'][word_id] = [word, 0]
            # create or increment word in this doc's word list
            # later the count will be replaced by tfidf
            if word_id in doc_words:
                doc_words[word_id] += 1
            else:
                doc_words[word_id] = 1
        # update corpus frequencies - each unique word in this doc adds 1
        for word in set(text):
            word_id = vocab['index'][word]
            vocab['words'][word_id][1] = vocab['words'][word_id][1] + 1
        timings['vocab'] += time.time()-t1
        # max_freq is being stored so we don't have to recalculate for the tfidf of every word
        try:
            max_freq = max(doc_words.values())
        except:
            print(f"Problem with doc: {filepath}")
            return None
        return [os.path.basename(filepath), doc_words, max_freq]

def remove_low_frequency_words(vocab, corpus, min_frequency=2):
    removed_words = set()
    removed_word_ids = set()
    for word_id in vocab['words'].copy():
        word, freq = vocab['words'][word_id]
        if freq < min_frequency:
            removed_words.add(word)
            removed_word_ids.add(word_id)
            del vocab['words'][word_id]
            del vocab['index'][word]
    for doc in corpus:
        words_to_remove = removed_word_ids.intersection(set(doc[1]))
        for word_id in words_to_remove:
            del doc[1][word_id]
    return vocab, corpus, removed_words

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

def save_project(vocab, corpus, output_folder_path):
    corpus_sub_folder = os.path.join(output_folder_path, 'corpus')
    pathlib.Path(corpus_sub_folder).mkdir(parents=True, exist_ok=True)
    for doc in corpus:
        filename = doc[0] + ".pickle"
        file_path = os.path.join(corpus_sub_folder, filename)
        with open(file_path, 'wb') as out_file:
            pickle.dump(doc, out_file)

    vocab_file_name = 'vocab.pickle'
    vocab_file_path = os.path.join(output_folder_path, vocab_file_name)
    with open(vocab_file_path, 'wb') as out_file:
        pickle.dump(vocab, out_file)

def load_project(input_folder_path, num_docs=None, random_files=False):
    """ num_docs won't make sense for tfidf, since vocab IDF was based on whole corpus """
    corpus_path = os.path.join(input_folder_path, 'corpus')
    corpus = []
    i = 0
    filenames = os.listdir(corpus_path)
    if random_files and num_docs > 0:
        filenames = random.sample(filenames, num_docs)
    for filename in filenames:
        i += 1
        if num_docs is not None and i > num_docs:
            break
        with open(os.path.join(corpus_path, filename), 'rb') as in_file:
            corpus.append(pickle.load(in_file))
    vocab_file_name = 'vocab.pickle'
    vocab_file_path = os.path.join(input_folder_path, vocab_file_name)
    with open(vocab_file_path, 'rb') as in_file:
        vocab = pickle.load(in_file)
    return vocab, corpus

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

def peek_doc(doc, vocabulary):
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
            bands_hashes.append(hash(tuple(band)))
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
                # Find if doc1 and doc2 have same hash value for any band
                for k in range(LSH_length):
                    if doc1[1][k] == doc2[1][k]:
                        #doc1_matches.append(doc2[0])
                        doc1_matches.append(j)
        #candidates[doc1[0]] = doc1_matches
        candidates[i] = doc1_matches
    return candidates

def group_candidates(candidates):
    # Find groups (clusters) of documents that are all similar to each other
    grouped_candidates = {}
    group_id = 0
    # Once we consider a doc, we do not want to consider it again
    # A document can only belong to one group (cluster)
    # After we have included a document in a cluster, add it to the exclusion_list so it won't be considered for future clusters
    exclusion_list = []
    for doc1 in list(candidates.keys()):
        if doc1 not in exclusion_list:
            docs_similar_to_doc1 = candidates[doc1]
            to_check = [x for x in docs_similar_to_doc1 if x not in exclusion_list]
            doc1_group = [doc1]
            # Check if doc_i that is similar to doc1 is also similar to all other docs in the cluster
            for doc_i in to_check:
                for clustered_doc in doc1_group:
                    # If these document appear in each other's similarity lists, add them to group
                    if doc_i in candidates[clustered_doc]:
                        doc1_group.append(doc_i)
                        # Remove those documents from pool
                        exclusion_list.append(doc_i)            
            grouped_candidates[group_id] = list(set(doc1_group))
            exclusion_list.append(doc1)
            group_id += 1
    return grouped_candidates

def top_group_words(grouped_candidates, corpus, vocabulary):
    new_group_obj = []
    # A word can only be associated with one cluster - first come, first served
    word_exclusion_list = []
    for group_num in grouped_candidates:
        group_docs = grouped_candidates[group_num]
        group_words = []
        for doc in group_docs:
            # Add largest tfidf word from each doc that is not already used, up to 10 words
            if len(group_words) < 10:
                added = False
                i = -1
                while not added:
                    largest_tfidf_word = vocabulary['words'][sorted(corpus[0][1], key=corpus[0][1].get)[i]][0]
                    if largest_tfidf_word not in word_exclusion_list:
                        group_words.append(largest_tfidf_word)
                        word_exclusion_list.append(largest_tfidf_word)
                        added = True
                    else:
                        i -= 1
        new_group_obj.append((group_docs, group_words))
    return new_group_obj

def get_doc_title_from_filename(filename, docs_dir):
    path = os.path.join(docs_dir, filename)
    doc = json.load(open(path))
    return doc['metadata']['title']

def lookup_word(word_id, vocabulary):
    return vocabulary['words'][word_id][0]

def plot_clusters(X,labels,num_docs):
    seaborn.set(rc={'figure.figsize':(15,15)})

    palette = seaborn.hls_palette(len(set(labels)), l=.4, s=.9)

    seaborn.scatterplot(X[0], X[1], hue=labels, legend='full', palette=palette);
    pyplot.title(f'Clusters with {num_docs} Documents');
    pyplot.xlabel('PCA Component 1');
    pyplot.ylabel('PCA Component 2');
    pyplot.show()
    return

def plot_k(K,WCSSE,optimal_k,num_docs):

    pyplot.rcParams["figure.figsize"] = (20,7)

    pyplot.plot(K, WCSSE, color='b');
    pyplot.axvline(x=optimal_k, color='r');
    pyplot.plot([K[0],K[-1]], [WCSSE[0],WCSSE[-1]], '--g');
    pyplot.title(f'Using {num_docs} Documents');
    pyplot.xlabel('k');
    pyplot.ylabel('WCSSE');
    pyplot.show()
    return

def reduce_to_2d(cluster_results, docs, vocab, k):
    # Use num_docs random words to represent docs, then PCA
    new_doc_length = min(len(docs), len(vocab['words']))
    random_words = set([])
    while len(random_words) < new_doc_length:
        random_words.update([random.randint(0, len(vocab['words'])-1)])
    X = []
    labels = []
    for i in range(len(cluster_results.clusters)):
        cluster = cluster_results.clusters[i]
        for j in range(len(cluster)):
            doc = cluster[j]
            row = numpy.zeros(shape=len(random_words))
            for m in range(len(random_words)):
                if m in doc[1].keys():
                    row[m] = doc[1][m]
                else:
                    row[m] = 0
            X.append(row)
            labels.append(i)
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    print(f'dim {k} -> 2 info preserved: {sum(pca.explained_variance_ratio_)}')
    return numpy.transpose(X), labels