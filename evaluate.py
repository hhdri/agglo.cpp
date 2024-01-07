import random

import numpy as np
from sklearn.cluster import AgglomerativeClustering

# load glove embeddings
vocab_size = 10_000
file_path = '/home/majid/repos/agglo.cpp/glove/glove.6B.300d.txt'
embeddings = []
vocab = []
with open(file_path, 'r') as f:
    for line in f:
        line = line.strip().split()
        vocab.append(line[0])
        embeddings.append(np.array(line[1:], dtype=np.float32))
        if len(vocab) == vocab_size:
            break
embeddings = np.array(embeddings)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1).reshape(-1, 1)
print('embeddings shape: ', embeddings.shape)

# load predictions
pred_cpp = []
with open('/home/majid/repos/agglo.cpp/clusters.txt', 'r') as f:
    for line in f:
        pred_cpp.append(int(line.split()[1]))
# random.shuffle(predictions)
pred_cpp = np.array(pred_cpp)

cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=1 - 0.6, metric='cosine', linkage='average')
pred_python = cluster.fit_predict(embeddings)

clusters_cpp = [[] for _ in range(max(pred_cpp) + 1)]
for i, c in enumerate(pred_cpp):
    clusters_cpp[c].append(vocab[i])

clusters_python = [[] for _ in range(max(pred_python) + 1)]
for i, c in enumerate(pred_python):
    clusters_python[c].append(vocab[i])

clusters_cpp = set([tuple(sorted(c)) for c in clusters_cpp])
clusters_python = set([tuple(sorted(c)) for c in clusters_python])

cluster_map_cpp = {word: words for cluster, words in enumerate(clusters_cpp) for word in words}
cluster_map_python = {word: words for cluster, words in enumerate(clusters_python) for word in words}

# number of words with different cluster in cpp and python
count = 0
disagreements = set()
for word in vocab:
    if cluster_map_cpp[word] != cluster_map_python[word]:
        count += 1
        disagreements.add((cluster_map_cpp[word], cluster_map_python[word]))
print('number of words with different cluster in cpp and python: ', count)
print(len(disagreements))

for c in disagreements:
    print(c[0])
    print(c[1])
    print('-------------------')