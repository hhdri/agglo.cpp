//
// Created by Majid Hajiheidari on 12/27/23.
//

#ifndef UNTITLED2_AGGLOMERATIVECLUSTERING_H
#define UNTITLED2_AGGLOMERATIVECLUSTERING_H

#include <unordered_set>
#include <memory>

using std::vector;

class Cluster {
public:
    std::unique_ptr<float[]> embedding;
    std::unordered_set<int> objects;

    Cluster(std::unordered_set<int> objects, const float *embeddings, int embeddingDim);

    // Move constructor
    Cluster(Cluster &&other) noexcept
            : objects(std::move(other.objects)), embedding(std::move(other.embedding)) {}

    // Move assignment operator
    Cluster &operator=(Cluster &&other) noexcept {
        if (this != &other) {
            objects = std::move(other.objects);
            embedding = std::move(other.embedding);
        }
        return *this;
    }

    // Delete copy constructor and copy assignment operator
    Cluster(const Cluster &) = delete;

    Cluster &operator=(const Cluster &) = delete;
};

class ClusterPair {
public:
    int cluster1Id;
    int cluster2Id;
    double similarity;
};

class AgglomerativeClustering {
public:
    int searchK;
    int embeddingDim;
    faiss::idx_t vocabSize;
    float *embeddings;
    float threshold;

    AgglomerativeClustering(float *embeddings, int embeddingDim, faiss::idx_t vocabSize, int searchK, float threshold);

    vector<Cluster> agglomerativeClustering();

private:
    void searchFaiss(int size, float *clusterEmbeddings, float *searchDistances, faiss::idx_t *searchIndices) const;

    vector<ClusterPair> buildClusterPairs(unsigned long numClusters, const float *searchDistances,
                                          const faiss::idx_t *searchIndices) const;

    float* buildClusterEmbeddings(const vector<Cluster> *clusters) const;

    vector<Cluster> populateNewClusters(const vector<ClusterPair> &clusterPairs, const vector<Cluster> *clusters);

    vector<Cluster> mergeClusters(vector<Cluster> *clusters);

    vector<Cluster> buildSingletonClusters();

};

Cluster::Cluster(std::unordered_set<int> objects, const float *embeddings, int embeddingDim)
        : objects(std::move(objects)), embedding(std::make_unique<float[]>(embeddingDim)) {
    auto clusterSize = static_cast<float>(this->objects.size());
    std::fill_n(embedding.get(), embeddingDim, 0.0f);

    for (int object: this->objects) {
        for (int i = 0; i < embeddingDim; i++) {
            embedding[i] += embeddings[object * embeddingDim + i] / clusterSize;
        }
    }
}

AgglomerativeClustering::AgglomerativeClustering(float *embeddings, int embeddingDim, faiss::idx_t vocabSize,
                                                 int searchK, float threshold)
        : embeddings(embeddings), embeddingDim(embeddingDim), vocabSize(vocabSize), searchK(searchK),
          threshold(threshold) {}

void
AgglomerativeClustering::searchFaiss(int size, float *clusterEmbeddings, float *searchDistances,
                                     faiss::idx_t *searchIndices) const {
    faiss::IndexFlatIP index(embeddingDim);
    index.add(size, clusterEmbeddings);
    index.search(size, clusterEmbeddings, searchK, searchDistances, searchIndices);
}

vector<ClusterPair>
AgglomerativeClustering::buildClusterPairs(unsigned long numClusters, const float *searchDistances,
                                           const faiss::idx_t *searchIndices) const {
    vector<ClusterPair> clusterPairs;
    for (int i = 0; i < numClusters; i++) {
        for (int j = 0; j < searchK; j++) {
            int cluster1Id = i;
            int cluster2Id = (int) searchIndices[i * searchK + j];
            if (searchDistances[i * searchK + j] > threshold & cluster1Id < cluster2Id) {
                clusterPairs.emplace_back(ClusterPair{cluster1Id, cluster2Id, searchDistances[i * searchK + j]});
            }
        }
    }
    std::sort(clusterPairs.begin(), clusterPairs.end(),
              [](const ClusterPair &a, const ClusterPair &b) { return a.similarity > b.similarity; });

    return clusterPairs;
}

float* AgglomerativeClustering::buildClusterEmbeddings(const vector<Cluster> *clusters) const {
    auto numClusters = clusters->size();
    auto *clusterEmbeddings = new float[numClusters * embeddingDim];
    for (int i = 0; i < numClusters; i++) {
        std::copy(
                clusters->at(i).embedding.get(),
                clusters->at(i).embedding.get() + embeddingDim,
                clusterEmbeddings + i * embeddingDim
        );
    }
    return clusterEmbeddings;
}

vector<Cluster> AgglomerativeClustering::populateNewClusters(const vector<ClusterPair> &clusterPairs,
                                                             const vector<Cluster> *clusters) {
    auto newClusters = vector<Cluster>();
    auto numClusters = clusters->size();
    vector<bool> isClusterUsed(numClusters, false);

    int numMerged = 0;
    for (const ClusterPair &clusterPair: clusterPairs) {
        if (isClusterUsed[clusterPair.cluster1Id] || isClusterUsed[clusterPair.cluster2Id]) {
            continue;
        }
        isClusterUsed[clusterPair.cluster1Id] = true;
        isClusterUsed[clusterPair.cluster2Id] = true;
        std::unordered_set<int> objects;
        objects.insert(clusters->at(clusterPair.cluster1Id).objects.begin(),
                       clusters->at(clusterPair.cluster1Id).objects.end());
        objects.insert(clusters->at(clusterPair.cluster2Id).objects.begin(),
                       clusters->at(clusterPair.cluster2Id).objects.end());
        newClusters.emplace_back(objects, embeddings, embeddingDim);
        numMerged++;
    }
    int numNotMerged = 0;
    for (int i = 0; i < numClusters; i++) {
        if (!isClusterUsed[i]) {
            newClusters.emplace_back(clusters->at(i).objects, embeddings, embeddingDim);
            numNotMerged++;
        }
    }
    std::cout << "Num clusters: " << newClusters.size() << " (" << numMerged << " merged, " << numNotMerged
              << " not merged)"
              << std::endl;
    return newClusters;
}

vector<Cluster> AgglomerativeClustering::mergeClusters(vector<Cluster> *clusters) {
    auto num_clusters = clusters->size();
//    auto *clusterEmbeddings = new float[num_clusters * embeddingDim];
    auto *clusterEmbeddings = buildClusterEmbeddings(clusters);

    auto *searchIndices = new faiss::idx_t[num_clusters * searchK];
    auto *searchDistances = new float[num_clusters * searchK];
    searchFaiss((int) num_clusters, clusterEmbeddings, searchDistances, searchIndices);
    delete[] clusterEmbeddings;

    auto clusterPairs = buildClusterPairs(num_clusters, searchDistances, searchIndices);
    delete[] searchIndices;
    delete[] searchDistances;

    auto newClusters = populateNewClusters(clusterPairs, clusters);

    return newClusters;
}

vector<Cluster> AgglomerativeClustering::buildSingletonClusters() {
    vector<Cluster> clusters;
    for (int i = 0; i < vocabSize; i++) {
        std::unordered_set<int> objects;
        objects.insert(i);
        clusters.emplace_back(objects, embeddings, embeddingDim);
    }
    return clusters;
}

vector<Cluster> AgglomerativeClustering::agglomerativeClustering() {
    auto clusters = buildSingletonClusters();

    for (int i = 0;; i++) {
        std::cout << "Iteration " << i << std::endl;
        auto newClusters = mergeClusters(&clusters);
        if (newClusters.size() == clusters.size()) {
            break;
        }
        clusters = std::move(newClusters);
    }

    return clusters;
}


#endif //UNTITLED2_AGGLOMERATIVECLUSTERING_H
