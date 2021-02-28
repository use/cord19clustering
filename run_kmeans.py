import random
import time
from pprint import pp

import kmeans
import library

if __name__ == '__main__':
    t0 = time.time()
    docs_dir = '../input/CORD-19-research-challenge/document_parses/pdf_json'

    t = time.time()
    num_docs = 5000
    vocab, docs = library.load_project('data_after_removing_words', num_docs, random_files=True)
    print(f"loaded {num_docs:,} docs {time.time()-t:.2f}")

    t = time.time()
    sub_corpus_freqs = kmeans.sub_corpus_frequencies(docs)
    print(f"recalc frequencies {time.time()-t:.2f}")

    results = kmeans.find_clusters(docs, 6)

    print(
        'sorted cluster sizes:',
        sorted([len(cluster) for cluster in results.clusters], reverse=True),
        f"(required {results.iterations} iterations)"
    )
    pp(kmeans.timings, indent=1)
    clusters = []
    for cluster in results.clusters:
        clusters.append({
            'cluster': cluster,
            'length': len(cluster),
            'common_words': kmeans.common_words_in_cluster(cluster, sub_corpus_freqs, vocab),
        })
    clusters.sort(key=lambda c: -c['length'])
    for index, cluster in enumerate(clusters):
        print(f"------- Cluster {index} -------")
        print(f"Size: {cluster['length']:,}")
        print(f"Defining words:")
        print("  - " + ", ".join([f"{word[0]} {round(word[2]*100)}% (+{round(word[1]*100)}%)" for word in cluster['common_words'][:10]]))
        print('5 random papers')
        sample = random.sample(cluster['cluster'], 5)
        for doc in sample:
            print(f"  - {library.get_doc_title_from_filename(doc[0], docs_dir)}")
            print(kmeans.doc_sorted_tfidf_words(doc, vocab['words'])[:10])

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
