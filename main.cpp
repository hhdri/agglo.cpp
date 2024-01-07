#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <unordered_set>
#include <algorithm>

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
        auto norm = std::sqrt(normSquared);
        for (int j = 0; j < embeddingDim; j++) {
            embeddings[i * embeddingDim + j] /= norm;
        }
    }
    file.close();
}

int main(int argc, char* argv[]) {
    const int EMBEDDING_DIM = 300;
    const int VOCAB_SIZE = 10'000;
    auto *embeddings = new float[VOCAB_SIZE * EMBEDDING_DIM];
    auto *vocab = new string[VOCAB_SIZE];
    const int searchK = 2;

    loadPartition({argv[1], VOCAB_SIZE}, 0, EMBEDDING_DIM,
                  embeddings, vocab);

////    print out 1000th word and its embedding
//    cout << vocab[1000] << endl;
//    for (int i = 0; i < EMBEDDING_DIM; i++) {
//        cout << embeddings[1000 * EMBEDDING_DIM + i] << " ";
//    }

    std::vector<Cluster> clusters;
    auto clustering = AgglomerativeClustering(embeddings, EMBEDDING_DIM, VOCAB_SIZE, searchK, 0.6);
    clusters = clustering.runAlgorithm();

    // Sort clusters based on size
    std::sort(clusters.begin(), clusters.end(),
              [](const Cluster &a, const Cluster &b) { return a.objects.size() > b.objects.size(); });

    vector<int> clusterIds(VOCAB_SIZE, -1);
    for (int i = 0; i < clusters.size(); i++) {
        for (int object: clusters[i].objects) {
            clusterIds[object] = i;
        }
    }

    // Write clusters to file
    std::ofstream file("/home/majid/repos/agglo.cpp/clusters.txt");
    for (int i = 0; i < VOCAB_SIZE; i++) {
        file << vocab[i] << " " << clusterIds[i];
        if (i < VOCAB_SIZE - 1)
            file << endl;
    }
    file.close();

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
}
