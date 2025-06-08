import sys

import numpy as np
from sklearn.cluster import AgglomerativeClustering

glove_path = sys.argv[1]
VOCAB_SIZE=int(sys.argv[2])
EMBEDDING_DIM=int(sys.argv[3])

embeddings = np.zeros((VOCAB_SIZE, EMBEDDING_DIM), dtype=np.float32)
vocab = []

with open(glove_path) as f:
    for line_no in range(VOCAB_SIZE):
        line = f.readline().split()
        vocab.append(line[0])
        for i in range(EMBEDDING_DIM):
            embeddings[line_no, i] = float(line[i+1])

embeddings = embeddings / np.expand_dims(np.linalg.norm(embeddings, axis=1), axis=1)

clustering = AgglomerativeClustering(
    n_clusters=None,
    metric='cosine',
    distance_threshold=100000,
    #compute_full_tree=True,
    linkage='average',
)
clustering.fit(embeddings)

def get_number_repr(node):
    if isinstance(node, int):
        return node
    return min(get_number_repr(node[0]), get_number_repr(node[1]))

nodes = {i:i for i in range(VOCAB_SIZE)}
for idx_, (left, right) in enumerate(clustering.children_):
    idx = idx_ + VOCAB_SIZE
    lc, rc = nodes[left], nodes[right]
    nodes[idx] = (min(lc, rc, key=get_number_repr), max(lc, rc, key=get_number_repr))
    del nodes[left], nodes[right]

print(nodes[2 * VOCAB_SIZE - 2])
