import random
import time
from dataclasses import dataclass
from pprint import pp
from typing import Dict, List, Tuple

Wordlist = Dict[int, float]
Doc = Tuple[str, Wordlist, float]
VocabWords = Dict[int, Tuple[str, float]]

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
        print(f"k={k}, iteration {iterations} ({(time.time()-iteration_timer):.2f}s) {lengths} skipped {skipped}")

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
        word_list.append((vocab['words'][word_id][0], frequency_diff, cluster_freq))
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

def optimal_k_WCSSEv2(K: List[int], WCSSE: List[float], threshold: float):
    greatest_dist = -1
    selected_k = -1
    endpoint_1 = (K[0], WCSSE[0])
    endpoint_2 = (K[-1], WCSSE[-1])
    results = []
    for i in range(len(K)):
        k = K[i]
        wcsse = WCSSE[i]
        dist = distance_from_line_to_point(endpoint_1, endpoint_2, (k, wcsse))
        results.append((k, wcsse, dist))
        if dist > greatest_dist:
            selected_k = k
            greatest_dist = dist
    print("WCSSE distance calculations:")
    pp(results)
    return selected_k

def doc_sorted_tfidf_words(doc: Doc, vocab_words: VocabWords) -> List[Tuple[str, float]]:
    words = [(vocab_words[word][0], round(doc[1][word], 3)) for word in doc[1]]
    return sorted(words, key=lambda word: -word[1])

def distance_from_line_to_point(endpoint_1: Tuple[float, float], endpoint_2: Tuple[float, float], point: Tuple[float, float]) -> float:
    "this uses the equation for finding the distance between a point and a line"
    x, y = 0, 1
    numerator = abs((endpoint_2[x] - endpoint_1[x]) * (endpoint_1[y] - point[y]) - (endpoint_1[x] - point[x]) * (endpoint_2[y] - endpoint_1[y]))
    denominator = ((endpoint_2[x] - endpoint_1[x]) ** 2 + (endpoint_2[y] - endpoint_1[y]) ** 2) ** (1/2)
    return numerator / denominator
