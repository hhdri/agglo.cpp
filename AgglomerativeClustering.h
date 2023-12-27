//
// Created by Majid Hajiheidari on 12/27/23.
//

#ifndef UNTITLED2_AGGLOMERATIVECLUSTERING_H
#define UNTITLED2_AGGLOMERATIVECLUSTERING_H

#include <unordered_set>
#include <memory>

class Cluster {
public:
    std::unique_ptr<float[]> embedding;
    std::unordered_set<int> objects;

    Cluster(std::unordered_set<int> objects, const float *embeddings, int embeddingDim);

    // Move constructor
    Cluster(Cluster&& other) noexcept
            : objects(std::move(other.objects)), embedding(std::move(other.embedding)) {}

    // Move assignment operator
    Cluster& operator=(Cluster&& other) noexcept {
        if (this != &other) {
            objects = std::move(other.objects);
            embedding = std::move(other.embedding);
        }
        return *this;
    }

    // Delete copy constructor and copy assignment operator
    Cluster(const Cluster&) = delete;
    Cluster& operator=(const Cluster&) = delete;
};

class ClusterPair {
public:
    int cluster1Id;
    int cluster2Id;
    double similarity;
};

Cluster::Cluster(std::unordered_set<int> objects, const float *embeddings, int embeddingDim)
        : objects(std::move(objects)), embedding(std::make_unique<float[]>(embeddingDim)) {
    auto clusterSize = static_cast<float>(this->objects.size());
    std::fill_n(embedding.get(), embeddingDim, 0.0f);

    for (int object : this->objects) {
        for (int i = 0; i < embeddingDim; i++) {
            embedding[i] += embeddings[object * embeddingDim + i] / clusterSize;
        }
    }
}


#endif //UNTITLED2_AGGLOMERATIVECLUSTERING_H
