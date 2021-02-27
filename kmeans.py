import csv
import random
import time
from dataclasses import dataclass
from pprint import pp
from typing import Dict, List, Tuple

import library

Wordlist = Dict[int, float]
Doc = Tuple[str, Wordlist, float]

@dataclass
class ClusterResults:
    clusters: List[List[Doc]]
    centroids: List[Doc]
    iterations: int
    wcsse: float

def item_distance_euclidian(a: Doc, b: Doc) -> float:
    a_words = set(a[1])
    b_words = set(b[1])
    a_only_words = a_words.difference(b_words)
    b_only_words = b_words.difference(a_words)
    shared_words = a_words.intersection(b_words)
    sum_a = sum([a[1][word] ** 2 for word in a_only_words])
    sum_b = sum([b[1][word] ** 2 for word in b_only_words])
    sum_c = sum([(a[1][word] - b[1][word]) ** 2 for word in shared_words])
    return (sum_a + sum_b + sum_c) ** (1/2)

norm_cache = {}
def item_distance_dot_product(a: Doc, b: Doc) -> float:
    shared_words = set(a[1]).intersection(b[1])
    similarity = sum(a[1][word] * b[1][word] for word in shared_words)
    similarity_normalized = similarity / (norm(a) * norm(b))
    return 1 - similarity_normalized

def norm(doc: Doc):
    if doc[0] in norm_cache:
        return norm_cache[doc[0]]
    freqs = doc[1]
    norm = sum(freqs[word] ** 2 for word in freqs) ** (1/2)
    norm_cache[doc[0]] = norm
    return norm

def find_centroid(cluster: List[Doc]) -> Doc:
    t = time.time()
    all_words = set()
    for doc in cluster:
        all_words.update(doc[1])
    result = dict.fromkeys(all_words)
    for word in result:
        result[word] = 0
    for doc in cluster:
        for word in doc[1]:
            result[word] += doc[1][word]
    threshold = .05
    for word in set(result):
        avg = result[word] / len(cluster)
        if avg > threshold:
            result[word] = avg
        else:
            del result[word]
    print(f"centroid length: {len(result)} ({(time.time() - t):.2f}s)")
    return ('centroid-'+str(random.randint(0, 999999)), result, 0)

timings = {
    'centroids': 0.0,
    'assign': 0.0,
    'dot product': 0.0,
}
def find_clusters(items: List[Doc], k: int):
    centroids = None
    old_clusters = [[] for x in range(k)]

    # index 0: index of prev cluster
    # index 1: stream of times landed in the same cluster
    item_meta: List[Tuple[int, int]] = [(-1, 0) for _ in range(len(items))]
    streak_threshold = 4 # after this many times in the same cluster, let's stop checking this item
    iterations = 0
    while True:
        iterations += 1
        new_clusters = [[] for x in range(k)]

        if not centroids:
            # select k objects at random for the first centroids
            centroids = random.sample(items, k)

        iteration_timer = time.time()
        t = time.time()
        # assign objects to clusters based on closest centroid
        skipped = 0
        for item_index, item in enumerate(items):
            lowest_distance = -1
            selected_index = -1
            
            prev_index, streak = item_meta[item_index]
            if streak > streak_threshold:
                selected_index = prev_index
                skipped += 1
            else:
                # find closest centroid to this object
                for index, centroid in enumerate(centroids):
                    t1 = time.time()
                    distance = item_distance_dot_product(item, centroid)
                    timings['dot product'] += time.time() - t1
                    if lowest_distance == -1 or distance < lowest_distance:
                        lowest_distance = distance
                        selected_index = index
                if prev_index == selected_index:
                    streak += 1
                else:
                    streak = 0
                item_meta[item_index] = (selected_index, streak)
            new_clusters[selected_index].append(item)
        timings['assign'] += time.time() - t

        lengths = [len(cluster) for cluster in new_clusters]
        print(f"iteration {iterations} ({(time.time()-iteration_timer):.2f}s) {lengths} skipped {skipped}")

        if new_clusters == old_clusters:
            # the following two lines can be used to add the centroids at index 0, so you can graph them
            # for index, cluster in enumerate(new_clusters):
            #     cluster.insert(0, centroids[index])
            t = time.time()
            wcsse = 0
            for index, cluster in enumerate(new_clusters):
                for doc in cluster:
                    wcsse += (item_distance_dot_product(doc, centroids[index])) ** 2
            print(f"wcsse ({(time.time()-t):.2f}s)")
            print(f"wcsse total: {wcsse:,.2f}")
            return ClusterResults(
                clusters = new_clusters,
                iterations = iterations,
                centroids = centroids,
                wcsse = wcsse,
            )
        t = time.time()
        centroids = [find_centroid(cluster) for cluster in new_clusters]
        timings['centroids'] += time.time() - t
        old_clusters = new_clusters

def common_words_in_cluster(items: List[Doc], corpus_freqs: Dict[int, int]):
    cluster_words = set()
    for doc in items:
        cluster_words.update(doc[1])
    cluster_freqs = {word: 0 for word in cluster_words}
    for doc in items:
        for word in doc[1]:
            cluster_freqs[word] += 1
    word_list = []
    for word_id in cluster_freqs:
        cluster_freq = cluster_freqs[word_id] / len(items)
        corpus_freq = corpus_freqs[word_id]
        frequency_diff = round(cluster_freq - corpus_freq, 2)
        word_list.append((vocab['words'][word_id][0], frequency_diff))
    word_list.sort(key=lambda word: -word[1])
    return word_list

def sub_corpus_frequencies(items: List[Doc]) -> Dict[int, int]:
    words_set = set()
    for doc in items:
        words_set.update(doc[1])
    freqs = dict.fromkeys(words_set)
    for word in freqs:
        freqs[word] = 0
    for doc in items:
        for word in doc[1]:
            freqs[word] += 1
    for word in freqs:
        freqs[word] = freqs[word] / len(items)
    return freqs

def optimal_k_WCSSE(K: List[int], WCSSE: List[float], threshold: float):
    for i in range(1,len(WCSSE)):
        if WCSSE[i-1]-WCSSE[i] < threshold:
            return K[i-1]
    return len(WCSSE)

if __name__ == '__main__':
    t0 = time.time()
    docs_dir = '../input/CORD-19-research-challenge/document_parses/pdf_json'

    t = time.time()
    num_docs = 50000
    vocab, docs = library.load_project('data_after_removing_words', num_docs, random_files=True)
    print(f"loaded {num_docs:,} docs {time.time()-t:.2f}")

    t = time.time()
    sub_corpus_freqs = sub_corpus_frequencies(docs)
    print(f"recalc frequencies {time.time()-t:.2f}")

    # create the clusters multiple times to compare
    kmax = 10
    K = []
    WCSSE = []
    for k in range(1, kmax + 1):
        K.append(k)
        results = find_clusters(docs, k)
        WCSSE.append(results.wcsse)
    scaled_WCSSE = [x/WCSSE[0] for x in WCSSE]
    optimal_k = optimal_k_WCSSE(K,scaled_WCSSE,0.05)
    results = find_clusters(docs, optimal_k)
    print(
        'sorted cluster sizes:',
        sorted([len(cluster) for cluster in results.clusters], reverse=True),
        f"(required {results.iterations} iterations)"
    )
    pp(timings, indent=1)
    clusters = []
    for cluster in results.clusters:
        clusters.append({
            'cluster': cluster,
            'length': len(cluster),
            'common_words': common_words_in_cluster(cluster, sub_corpus_freqs),
        })
    clusters.sort(key=lambda c: -c['length'])
    for index, cluster in enumerate(clusters):
        print(f"------- Cluster {index} -------")
        print(f"Size: {cluster['length']:,}")
        print(f"Defining words:")
        print("  - " + ", ".join([f"{word[0]} +{round(word[1]*100)}%" for word in cluster['common_words'][:10]]))
        print('5 random papers')
        sample = random.sample(cluster['cluster'], 5)
        for doc in sample:
            print(f"  - {library.get_doc_title_from_filename(doc[0], docs_dir)}")

    freqs_list = [(library.lookup_word(word, vocab), sub_corpus_freqs[word]) for word in sub_corpus_freqs]
    freqs_sample = random.sample(freqs_list, 5)
    freqs_top_10 = sorted(freqs_list, key=lambda word: word[1], reverse=True)[:10]

    print(f"------- Corpus Stats -------")
    print("10 most frequent words")
    print(freqs_top_10)
    print("Random sample of word frequencies")
    print(freqs_sample)
    lookup_words = ['coronavirus', 'covid19', 'government', 'policy', 'respiratory']
    print("Frequencies of some meaningful words:")
    print(", ".join([f"{word} {sub_corpus_freqs[vocab['index'][word]]}" for word in lookup_words]))

    print(f"total time: {time.time() - t0}")

    exit()
    # only save the results of the first attempt
    if i == 0:
        for index, cluster in enumerate(results.clusters):
            with open(f"cluster-{index+1}.csv", 'w', newline='') as outfile:
                writer = csv.writer(outfile)
                for item in cluster:
                    writer.writerow(item)
