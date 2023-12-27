//
// Created by Majid Hajiheidari on 12/27/23.
//

#ifndef UNTITLED2_AGGLOMERATIVECLUSTERING_H
#define UNTITLED2_AGGLOMERATIVECLUSTERING_H

#include <unordered_set>

template<int embeddingDim>
class Cluster {
public:
    float embedding[embeddingDim];
    std::unordered_set<int> objects;
    Cluster(std::unordered_set<int> objects, const float *embeddings);
};

class ClusterPair {
public:
    int cluster1Id;
    int cluster2Id;
    double similarity;
};

template<int embeddingDim>
Cluster<embeddingDim>::Cluster(std::unordered_set<int> objects, const float *embeddings) : objects(std::move(objects)) {
    auto clusterSize = (float) this->objects.size();
    for (float &i: embedding) {
        i = 0.0;
    }
    for (int object: this->objects) {
        for (int i = 0; i < embeddingDim; i++) {
            embedding[i] += embeddings[object * embeddingDim + i] / clusterSize;
        }
    }
}


#endif //UNTITLED2_AGGLOMERATIVECLUSTERING_H
