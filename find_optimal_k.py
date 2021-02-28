import time

import kmeans
import library

if __name__ == '__main__':
    t0 = time.time()
    docs_dir = '../input/CORD-19-research-challenge/document_parses/pdf_json'

    t = time.time()
    num_docs = 2000
    vocab, docs = library.load_project('data_after_removing_words', num_docs, random_files=True)
    print(f"loaded {num_docs:,} docs {time.time()-t:.2f}")

    t = time.time()
    sub_corpus_freqs = kmeans.sub_corpus_frequencies(docs)
    print(f"recalc frequencies {time.time()-t:.2f}")

    # create the clusters multiple times to compare
    kmax = 10
    K = []
    WCSSE = []
    for k in range(1, kmax + 1):
        K.append(k)
        results = kmeans.find_clusters(docs, k)
        WCSSE.append(results.wcsse)
    print(f"WCSSE: {WCSSE}")
    scaled_WCSSE = [x/WCSSE[0] for x in WCSSE]
    optimal_k = kmeans.optimal_k_WCSSE(K,scaled_WCSSE,0.05)
    optimal_k_v2 = kmeans.optimal_k_WCSSEv2(K,scaled_WCSSE,0.05)
    print(f"optimal k: {optimal_k}")
    print(f"optimal k (method 2): {optimal_k_v2}")
    print(f"total time: {time.time() - t0}")
