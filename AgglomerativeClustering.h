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

    AgglomerativeClustering(float *embeddings, int embeddingDim, faiss::idx_t vocabSize, int searchK);

    std::vector<Cluster> agglomerativeClustering();

private:
    void searchFaiss(int size, float *clusterEmbeddings, float *searchDistances, faiss::idx_t *searchIndices) const;

    void buildClusterPairs(unsigned long numClusters, const float *searchDistances,
                           const faiss::idx_t *searchIndices,
                           std::vector<ClusterPair> &clusterPairs) const;

    void buildClusterEmbeddings(const std::vector<Cluster> *clusters, float *clusterEmbeddings) const;

    void populateNewClusters(const std::vector<ClusterPair> &clusterPairs, const std::vector<Cluster> *clusters,
                             std::vector<Cluster> *newClusters);

    void mergeClusters(std::vector<Cluster> *clusters, std::vector<Cluster> *newClusters);

    void buildSingletonClusters(std::vector<Cluster> *clusters);

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
                                                 int searchK)
        : embeddings(embeddings), embeddingDim(embeddingDim), vocabSize(vocabSize), searchK(searchK) {}

void
AgglomerativeClustering::searchFaiss(int size, float *clusterEmbeddings, float *searchDistances,
                                     faiss::idx_t *searchIndices) const {
    faiss::IndexFlatIP index(embeddingDim);
    index.add(size, clusterEmbeddings);
    index.search(size, clusterEmbeddings, searchK, searchDistances, searchIndices);
}

void AgglomerativeClustering::buildClusterPairs(unsigned long numClusters, const float *searchDistances,
                                                const faiss::idx_t *searchIndices,
                                                std::vector<ClusterPair> &clusterPairs) const {
    for (int i = 0; i < numClusters; i++) {
        for (int j = 0; j < searchK; j++) {
            int cluster1Id = i;
            int cluster2Id = (int) searchIndices[i * searchK + j];
            if (searchDistances[i * searchK + j] > 0.95 & cluster1Id < cluster2Id) {
                clusterPairs.emplace_back(ClusterPair{cluster1Id, cluster2Id, searchDistances[i * searchK + j]});
            }
        }
    }
    std::sort(clusterPairs.begin(), clusterPairs.end(),
              [](const ClusterPair &a, const ClusterPair &b) { return a.similarity > b.similarity; });
}

void
AgglomerativeClustering::buildClusterEmbeddings(const std::vector<Cluster> *clusters, float *clusterEmbeddings) const {
    auto numClusters = clusters->size();
    for (int i = 0; i < numClusters; i++) {
        std::copy(
                clusters->at(i).embedding.get(),
                clusters->at(i).embedding.get() + embeddingDim,
                clusterEmbeddings + i * embeddingDim
        );
    }
}

void AgglomerativeClustering::populateNewClusters(const std::vector<ClusterPair> &clusterPairs,
                                                  const std::vector<Cluster> *clusters,
                                                  std::vector<Cluster> *newClusters) {
    auto numClusters = clusters->size();
    std::vector<bool> isClusterUsed(numClusters, false);

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
        newClusters->emplace_back(objects, embeddings, embeddingDim);
        numMerged++;
    }
    int numNotMerged = 0;
    for (int i = 0; i < numClusters; i++) {
        if (!isClusterUsed[i]) {
            newClusters->emplace_back(clusters->at(i).objects, embeddings, embeddingDim);
            numNotMerged++;
        }
    }
    std::cout << "Num clusters: " << newClusters->size() << " (" << numMerged << " merged, " << numNotMerged
              << " not merged)"
              << std::endl;
}

void
AgglomerativeClustering::mergeClusters(std::vector<Cluster> *clusters, std::vector<Cluster> *newClusters) {
    unsigned long num_clusters = clusters->size();
    auto *clusterEmbeddings = new float[num_clusters * embeddingDim];
    buildClusterEmbeddings(clusters, clusterEmbeddings);

    auto *searchIndices = new faiss::idx_t[num_clusters * searchK];
    auto *searchDistances = new float[num_clusters * searchK];
    searchFaiss(num_clusters, clusterEmbeddings, searchDistances, searchIndices);

    delete[] clusterEmbeddings;

    std::vector<ClusterPair> clusterPairs;
    buildClusterPairs(num_clusters, searchDistances, searchIndices, clusterPairs);

    delete[] searchIndices;
    delete[] searchDistances;

    populateNewClusters(clusterPairs, clusters, newClusters);
}

void AgglomerativeClustering::buildSingletonClusters(std::vector<Cluster> *clusters) {
    for (int i = 0; i < vocabSize; i++) {
        std::unordered_set<int> objects;
        objects.insert(i);
        clusters->emplace_back(objects, embeddings, embeddingDim);
    }
}

std::vector<Cluster> AgglomerativeClustering::agglomerativeClustering() {
    std::vector<Cluster> clusters, newClusters;
    buildSingletonClusters(&clusters);

    for (int i = 0;; i++) {
        std::cout << "Iteration " << i << std::endl;
        mergeClusters(&clusters, &newClusters);
        if (newClusters.size() == clusters.size()) {
            break;
        }
        clusters = std::move(newClusters);
        newClusters = std::vector<Cluster>();
    }

    return clusters;
}


#endif //UNTITLED2_AGGLOMERATIVECLUSTERING_H
