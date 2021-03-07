import os
import kmeans

num_words = 6
results_path = 'clusters'

filenames = os.listdir(results_path)
clusterings = []
for filename in filenames:
    if filename.endswith('.pickle'):
        clustering = kmeans.load_clusters(os.path.join(results_path, filename))
        clustering.sort(key=lambda cluster: - len(cluster['files']))
        clusterings.append(clustering)

num_clusters = len(clusterings[0])
num_docs = sum(len(cluster['files']) for cluster in clusterings[0])

for i, clustering in enumerate(clusterings):
    print(f"Clustering #{i+1} (k={num_clusters}, n={num_docs:,})")
    for cluster in clustering:
        print(
            '  ' +
            str(f"{len(cluster['files']):,}") + ': ' +
            str(", ".join(f"{word[0]} +{round(word[1]*100)}%" for word in cluster['common_words'][:num_words]))
        )
