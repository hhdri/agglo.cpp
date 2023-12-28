#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <unordered_set>
#include <algorithm>
#include <cmath>

#include <faiss/IndexFlat.h>

#include "AgglomerativeClustering.h"

using std::string, std::cout, std::endl;

struct Partition {
    string path;
    int size;
};

void loadPartition(const Partition &partition, int offset, int embeddingDim, float *embeddings, string *vocab) {
    std::ifstream file(partition.path);
    for (int i = offset; i < offset + partition.size; i++) {
        string line, word;
        std::getline(file, line);
        auto lineStream = std::istringstream(line);
        lineStream >> word;
        vocab[i] = word;

        float normSquared = 0.0;
        for (int j = 0; j < embeddingDim; j++) {
            float value;
            lineStream >> value;
            embeddings[i * embeddingDim + j] = value;
            normSquared += value * value;
        }
        float norm = sqrt(normSquared);
        for (int j = 0; j < embeddingDim; j++) {
            embeddings[i * embeddingDim + j] /= norm;
        }
    }
    file.close();
}

int main() {
    const int EMBEDDING_DIM = 50;
    const int VOCAB_SIZE = 50'000;
    auto *embeddings = new float[VOCAB_SIZE * EMBEDDING_DIM];
    auto *vocab = new string[VOCAB_SIZE];
    const int searchK = 20;

    loadPartition({"/Users/majid/Downloads/glove.6B/glove.6B.50d/glove.6B.50d.aa", VOCAB_SIZE}, 0, EMBEDDING_DIM,
                  embeddings, vocab);

    std::vector<Cluster> clusters;
    auto clustering = AgglomerativeClustering(embeddings, EMBEDDING_DIM, VOCAB_SIZE, searchK, 0.6);
    clusters = clustering.runAlgorithm();

    // Sort clusters based on size
    std::sort(clusters.begin(), clusters.end(),
              [](const Cluster &a, const Cluster &b) { return a.objects.size() > b.objects.size(); });

    std::vector<std::vector<string>> clusterObjects;

    for (const Cluster &cluster: clusters) {
        std::vector<string> objects;
        objects.reserve(cluster.objects.size());
        for (int object: cluster.objects) {
            objects.push_back(vocab[object]);
        }
        clusterObjects.push_back(objects);
    }


    delete[] embeddings;
    delete[] vocab;

    cout << "Done" << endl;

    // TODO: Cluster size limit
    // TODO: Search only on clusters who are candidates for merging (if no pairs in this iteration, skip)
}