import library
import time
from pprint import pp

t = time.time()
vocab, corpus = library.load_project('data', num_docs=64000)
print(f"Loaded project {time.time() - t}")
print(f"{(len(corpus)):,} docs")

print("Removing low frequency words")
length_1 = len(vocab['index'])
print(f"Starting vocab size: {length_1:,}")

avg_unique_words = sum([len(doc[1]) for doc in corpus]) / len(corpus)
print(f"Avg unique words per doc: {avg_unique_words:,}")

t = time.time()
vocab, corpus, removed_words = library.remove_low_frequency_words(vocab, corpus, min_frequency=2)
print(f"Removed words {time.time() - t}")

length_2 = len(vocab['index'])
print(f"Current vocab size: {length_2:,}")
print(f"Removed {(length_1 - length_2):,} words")

avg_unique_words = sum([len(doc[1]) for doc in corpus]) / len(corpus)
print(f"Avg unique words per doc: {avg_unique_words:,}")

pp(list(removed_words)[0:10])
pp(list(removed_words)[-10:-1])
