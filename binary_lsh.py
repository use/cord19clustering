import functools
import itertools
import json
import math
import multiprocessing
import os
import random
import time

import sympy

import library

docs_dir = '../input/CORD-19-research-challenge/document_parses/pdf_json'


def get_signature(doc, rands_a, rands_b, sig_size, modulus):
    words = doc[1]
    sig = [min([(rands_a[i] * word + rands_b[i]) % modulus for word in words])
           for i in range(sig_size)]
    return sig


def hash_bands(sig, num_bands):

    bands = []
    items_per_band = len(sig) // num_bands
    for index in range(num_bands):
        start = index * items_per_band
        stop = start + items_per_band
        items = sig[start:stop]
        bands.append((index, hash(tuple(items))))

    return bands


def find_candidates(docs_with_bands):
    groups = {}
    for doc_with_bands in docs_with_bands:
        doc, bands = doc_with_bands
        for index, band in bands:
            key = (index, band)
            if key in groups:
                groups[key].append(doc)
            else:
                groups[key] = [doc]
    groups = filter(lambda group: len(group) > 1, groups.values())
    return groups


def doc_similarity(doc_a, doc_b):
    a_words = set(doc_a[1])
    b_words = set(doc_b[1])
    return len(a_words.intersection(b_words)) / len(a_words.union(b_words))


def get_vocab_length(corpus):
    """ This is needed since the vocab object may not match the subsample of docs we're looking at """
    vocab = set()
    for doc in corpus:
        vocab.update(doc[1])
    return len(vocab)


def get_doc_title_from_filename(filename):
    path = os.path.join(docs_dir, filename)
    doc = json.load(open(path))
    return doc['metadata']['title']


def hash_doc(doc, a=[], b=[], sig_size=0, modulus=0, num_bands=0):
    return (doc, hash_bands(get_signature(doc, a, b, sig_size, modulus), num_bands))


def get_pair_sig(a, b):
    filename_a = a[0]
    filename_b = b[0]
    first = ''
    second = ''
    if filename_a < filename_b:
        first = filename_a
        second = filename_b
    else:
        first = filename_b
        second = filename_a
    return hash(first + '|' + second)


if __name__ == '__main__':
    timings = []

    t = time.time()
    vocab, corpus = library.load_project('data', 8000)
    timings.append(f"load project: {time.time()-t}")

    t = time.time()
    vocab_length = get_vocab_length(corpus)
    print(f"vocab length: {vocab_length}")
    timings.append(f"get_vocab_length: {time.time()-t}")

    NUM_BANDS = 10
    MODULUS = sympy.nextprime(vocab_length)
    SIGNATURE_SIZE = 100
    A = random.sample(range(1, 1500), SIGNATURE_SIZE)
    B = random.sample(range(1, 1500), SIGNATURE_SIZE)

    t = time.time()
    # serial approach
    # docs_with_bands = [(doc, hash_bands(get_signature(doc, A, B, SIGNATURE_SIZE, MODULUS), NUM_BANDS)) for doc in corpus]
    # parallel approach
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        docs_with_bands = pool.map(
            functools.partial(
                hash_doc,
                a=A,
                b=B,
                sig_size=SIGNATURE_SIZE,
                modulus=MODULUS,
                num_bands=NUM_BANDS
            ),
            corpus,
            chunksize=20
        )
    timings.append(f"docs_with_bands: {time.time()-t}")

    t = time.time()
    candidate_groups = find_candidates(docs_with_bands)
    timings.append(f"candidate_groups: {time.time()-t}")

    t = time.time()
    similarities = []
    processed_pair_sigs = set()
    non_unique_candidate_pairs = 0
    for group in candidate_groups:
        for pair in itertools.combinations(group, 2):
            non_unique_candidate_pairs += 1
            a, b = pair
            pair_sig = get_pair_sig(a, b)
            if a[0] == b[0]:
                print('how could this happen?')
                exit()
            if pair_sig in processed_pair_sigs:
                continue
            processed_pair_sigs.add(pair_sig)
            title_a = get_doc_title_from_filename(a[0])
            title_b = get_doc_title_from_filename(b[0])
            similarities.append((doc_similarity(a, b), title_a, title_b))
    similarities = sorted(similarities, key=lambda item: -item[0])
    print(f"{non_unique_candidate_pairs} non-unique candidate pairs")
    print(f"{len(similarities)} candidate pairs")
    timings.append(f"similarities: {time.time()-t}")

    top_x = 10
    print(f"Similarity of top {top_x} candidates, sorted")
    print("[similarity, doc1, doc2]")
    for item in similarities[0:top_x]:
        print(item)

    for timing in timings:
        print(timing)
