import csv
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
import time
from pprint import pp

import library

Wordlist = Dict[int, float]
Doc = Tuple[str, Wordlist, float]

@dataclass
class ClusterResults:
    clusters: List[List[Doc]]
    centroids: List[Doc]
    iterations: int

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

def item_distance_dot_product(a: Doc, b: Doc) -> float:
    shared_words = set(a[1]).intersection(b[1])
    return -sum(a[1][word] * b[1][word] for word in shared_words)

def find_centroid(cluster: List[Doc]) -> Doc:
    all_words = set()
    for doc in cluster:
        all_words.update(doc[1])
    result = {}
    doc_count = len(cluster)
    for word in all_words:
        word_sum = sum([doc[1][word] for doc in cluster if word in doc[1]])
        word_avg = word_sum / doc_count
        if word_avg > .05:
            result[word] = word_avg
    print(f"centroid length: {len(result)}")
    return ('centroid', result, 0)

timings = {
    'clusters': [],
    'centroids': 0,
    'assign': 0,
    'dot product': 0,
}
def find_clusters(items: List[Doc], k: int):
    centroids = None
    old_clusters = [[] for x in range(k)]

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
        inner_loop = 0
        for item in items:
            lowest_distance = -1
            selected_index = -1

            # find closest centroid to this object
            for index, centroid in enumerate(centroids):
                inner_loop += 1
                t1 = time.time()
                distance = item_distance_dot_product(item, centroid)
                timings['dot product'] += time.time() - t1
                if lowest_distance == -1 or distance < lowest_distance:
                    lowest_distance = distance
                    selected_index = index
            new_clusters[selected_index].append(item)
        timings['assign'] += time.time() - t

        print(f"iteration {iterations} {(time.time()-iteration_timer):.3f}")

        if new_clusters == old_clusters:
            # the following two lines can be used to add the centroids at index 0, so you can graph them
            # for index, cluster in enumerate(new_clusters):
            #     cluster.insert(0, centroids[index])
            return ClusterResults(
                clusters = new_clusters,
                iterations = iterations,
                centroids = centroids,
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

if __name__ == '__main__':
    t0 = time.time()
    items = []
    vocab, docs = library.load_project('data_after_removing_words', 500)

    sub_corpus_freqs = sub_corpus_frequencies(docs)

    # create the clusters multiple times to compare
    for i in range(1):
        results = find_clusters(docs, 10)
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
        for cluster in clusters:
            print(cluster['length'])
            print(cluster['common_words'][:10])
        print(time.time() - t0)
        exit()
        # only save the results of the first attempt
        if i == 0:
            for index, cluster in enumerate(results.clusters):
                with open(f"cluster-{index+1}.csv", 'w', newline='') as outfile:
                    writer = csv.writer(outfile)
                    for item in cluster:
                        writer.writerow(item)
