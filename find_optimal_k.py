import time

import kmeans
import library

if __name__ == '__main__':
    t0 = time.time()
    docs_dir = '../input/CORD-19-research-challenge/document_parses/pdf_json'

    t = time.time()
    num_docs = 160
    #vocab, docs = library.load_project('data_after_removing_words', num_docs, random_files=True)
    pickle_dir = 'C:\\Users\\Colin\\Documents\\SCHOOL_STUFF\\EWU\\W21\\CSCD_530\\Project\\Code\\data_after_removing_words'
    vocab, docs = library.load_project(pickle_dir, num_docs, random_files=True)
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
    optimal_k_v2 = kmeans.optimal_k_WCSSEv2(K,WCSSE)
    print(f"optimal k (method 2): {optimal_k_v2}")
    print(f"total time: {time.time() - t0}")

    results = kmeans.find_clusters(docs, optimal_k_v2)

    kmeans.plot_k(K,WCSSE,optimal_k_v2,num_docs)