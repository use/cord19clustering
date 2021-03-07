import library
import random
from pprint import pp
import kmeans
import time

vocab, docs = library.load_project('data_after_removing_words', num_docs=10000, random_files=False)

num_samples = 8000
overlaps = []
lengths = []
distances = []

dist_time = 0
for _ in range(num_samples):
    a, b = random.sample(docs, 2)
    intersection = set(a[1]).intersection(b[1])
    overlaps.append(len(intersection))
    lengths.append(len(a[1]))
    t = time.time()
    distances.append(kmeans.item_distance_dot_product(a, b, use_cache=True))
    dist_time += time.time() - t

average_overlap = sum(overlaps) / len(overlaps)
average_length = sum(lengths) / len(lengths)
# pp(docs[3])
print(f"average number of shared words: {average_overlap}")
print(f"average length: {average_length}")
print(f"highest distance: {max(distances)}")
print(f"lowest distance: {min(distances)}")
print(f"average distance: {sum(distances) / len(distances)}")
print(f"dist: {dist_time}")
