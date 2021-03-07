import pathlib
import pickle
import random
import time
from dataclasses import dataclass
from pprint import pp
from typing import Any, Dict, List, Tuple
from matplotlib import pyplot
from sklearn.decomposition import PCA
import multiprocessing
import functools

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
def item_distance_dot_product(a: Doc, b: Doc, use_cache=True) -> float:
    similarity = 0
    for word in a[1]:
        if word in b[1]:
            similarity += a[1][word] * b[1][word]
    norm_a = norm(a, use_cache=use_cache)
    norm_b = norm(b, use_cache=use_cache)
    if norm_a == 0:
        print(f"doc has norm zero: {a[0]}")
        return 1
    if norm_b == 0:
        print(f"doc has norm zero: {b[0]}")
        return 1
    similarity_normalized = similarity / (norm_a * norm_b)
    return 1 - similarity_normalized

def norm(doc: Doc, use_cache=True):
    if use_cache and doc[0] in norm_cache:
        return norm_cache[doc[0]]
    freqs = doc[1]
    norm = sum(freqs[word] ** 2 for word in freqs) ** (1/2)
    if use_cache:
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
        method = 2
        if method==1:
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

        if method==2:
            with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
                cluster_map = pool.map(
                    functools.partial(
                        closest_centroid,
                        centroids=centroids,
                    ),
                    items,
                    chunksize=200,
                )
            for item_index, cluster_index in enumerate(cluster_map):
                new_clusters[cluster_index].append(items[item_index])
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

def closest_centroid(item: Doc, centroids: List[Doc]) -> int:
    lowest_distance = -1
    selected_index = -1
    for index, centroid in enumerate(centroids):
        distance = item_distance_dot_product(item, centroid)
        if lowest_distance == -1 or distance < lowest_distance:
            lowest_distance = distance
            selected_index = index
    return selected_index

def common_words_in_cluster(items: List[Doc], corpus_freqs: Dict[int, int], vocab):
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
    return word_list[:100]

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

def optimal_k_WCSSEv2(K: List[int], WCSSE: List[float]):
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

def save_clusters(clusters: List[Dict[str, Any]], output_file_path: str):
    output = []
    for cluster in clusters:
        file_list = []
        for doc in cluster['cluster']:
            file_list.append(doc[0])
        output.append({
            'files': file_list,
            'common_words': cluster['common_words'],
        })
    with open(output_file_path, 'wb') as out_file:
        pickle.dump(output, out_file)

def load_clusters(input_file_path: str):
    with open(input_file_path, 'rb') as in_file:
        return pickle.load(in_file)

def plot_clusters(X,labels,num_docs):
    seaborn.set(rc={'figure.figsize':(15,15)})

    palette = seaborn.hls_palette(len(set(labels)), l=.4, s=.9)

    seaborn.scatterplot(X[0], X[1], hue=labels, legend='full', palette=palette);
    pyplot.title(f'Clusters with {num_docs} Documents');
    pyplot.xlabel('PCA Component 1');
    pyplot.ylabel('PCA Component 2');
    pyplot.show()
    return

def reduce_to_kd_2d(cluster_results, vocab, k):
    # Use k random words to represent docs, then PCA
    random_words = set([])
    while len(random_words) < k:
        random_words.update([random.randint(0, len(vocab['words'])-1)])
    X = []
    labels = []
    for i in range(len(cluster_results.clusters)):
        cluster = cluster_results.clusters[i]
        for j in range(len(cluster)):
            doc = cluster[j]
            row = numpy.zeros(shape=k)
            for m in range(k):
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

def plot_k(K: List[int], WCSSE: List[float], optimal_k: int, num_docs: int):

    pyplot.rcParams.update({"figure.figsize": (15,15), 'font.size':22})

    pyplot.plot(K, WCSSE, color='b', label='Total WCSSE at k')
    pyplot.axvline(x=optimal_k, color='r', label='Optimal k')
    pyplot.plot([K[0],K[-1]], [WCSSE[0],WCSSE[-1]], '--g', label='Guideline')
    pyplot.title(f'Optimal k with n = {num_docs}')
    pyplot.xlabel('k')
    pyplot.ylabel('WCSSE')
    pyplot.legend()
    pyplot.savefig(f'optimal_k_plot_n{num_docs}.png')
    pyplot.show()
    return
