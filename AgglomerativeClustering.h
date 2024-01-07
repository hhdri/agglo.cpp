#ifndef AGGLO_CPP_AGGLOMERATIVECLUSTERING_H
#define AGGLO_CPP_AGGLOMERATIVECLUSTERING_H

#include <unordered_set>
#include <memory>

using std::vector;

class Cluster {
public:
    std::unique_ptr<float[]> embedding;
    std::unordered_set<int> objects;
    bool mergable;

    Cluster(std::unordered_set<int> objects, const float *embeddings, int embeddingDim, bool mergable);

    // Move constructor
    Cluster(Cluster &&other) noexcept
            : objects(std::move(other.objects)), embedding(std::move(other.embedding)), mergable(other.mergable) {}

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

struct SubsetEmbeddings {
    int subsetSize;
    vector<int> subsetIndices;
    vector<float> embeddings;
};

class AgglomerativeClustering {
public:
    int searchK;
    int embeddingDim;
    faiss::idx_t vocabSize;
    float *embeddings;
    float threshold;

    AgglomerativeClustering(float *embeddings, int embeddingDim, faiss::idx_t vocabSize, int searchK, float threshold);

    vector<Cluster> runAlgorithm();

private:
    class FaissSearchResult {
    public:
        float *distances;
        faiss::idx_t *indices;

        FaissSearchResult(int size, int searchK) {
            distances = new float[size * searchK];
            indices = new faiss::idx_t[size * searchK];
        }

        ~FaissSearchResult() {
            delete[] distances;
            delete[] indices;
        }
    };

    FaissSearchResult searchFaiss(const SubsetEmbeddings &subsetEmbeddings) const;

    vector<ClusterPair>
    buildClusterPairs(SubsetEmbeddings &subsetEmbeddings, const FaissSearchResult *searchResult) const;

    SubsetEmbeddings buildClusterEmbeddings(const vector<Cluster> *clusters) const;

    vector<Cluster> buildNewClusters(const vector<ClusterPair> &clusterPairs, const vector<Cluster> *clusters);

    vector<Cluster> mergeClusters(vector<Cluster> *clusters);

    vector<Cluster> getSingletonClusters();

};

Cluster::Cluster(std::unordered_set<int> objects, const float *embeddings, int embeddingDim, bool mergable)
        : objects(std::move(objects)), embedding(std::make_unique<float[]>(embeddingDim)), mergable(mergable) {
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

AgglomerativeClustering::FaissSearchResult
AgglomerativeClustering::searchFaiss(const SubsetEmbeddings &subsetEmbeddings) const {
    AgglomerativeClustering::FaissSearchResult result(subsetEmbeddings.subsetSize, searchK);
    faiss::IndexFlatIP index(embeddingDim);
    index.add(subsetEmbeddings.subsetSize, subsetEmbeddings.embeddings.data());
    index.search(subsetEmbeddings.subsetSize, subsetEmbeddings.embeddings.data(), searchK, result.distances,
                 result.indices);
    return result;
}

vector<ClusterPair> AgglomerativeClustering::buildClusterPairs(SubsetEmbeddings &subsetEmbeddings,
                                                               const AgglomerativeClustering::FaissSearchResult *searchResult) const {
    vector<ClusterPair> clusterPairs;
    for (int i = 0; i < subsetEmbeddings.subsetSize; i++) {
        for (int j = 0; j < searchK; j++) {
            int cluster1Id = subsetEmbeddings.subsetIndices[i];
            int cluster2Id = subsetEmbeddings.subsetIndices[(int) searchResult->indices[i * searchK + j]];
            float similarity = searchResult->distances[i * searchK + j];
            if (cluster1Id != cluster2Id & similarity > threshold) {
                clusterPairs.emplace_back(
                        ClusterPair{cluster1Id, cluster2Id, similarity});
            }
        }
    }
    std::sort(clusterPairs.begin(), clusterPairs.end(),
              [](const ClusterPair &a, const ClusterPair &b) { return a.similarity > b.similarity; });

    return clusterPairs;
}

SubsetEmbeddings AgglomerativeClustering::buildClusterEmbeddings(const vector<Cluster> *clusters) const {
    size_t numClusters = clusters->size();

    int numMergableClusters = 0;
    for (int i = 0; i < numClusters; i++)
        numMergableClusters += clusters->at(i).mergable;

    vector<float> clusterEmbeddings(numMergableClusters * embeddingDim);
    vector<int> subsetIndices(numMergableClusters);

    numMergableClusters = 0;
    for (int i = 0; i < numClusters; i++) {
        if (!clusters->at(i).mergable)
            continue;
        subsetIndices[numMergableClusters] = i;
        std::copy(
                clusters->at(i).embedding.get(),
                clusters->at(i).embedding.get() + embeddingDim,
                clusterEmbeddings.begin() + numMergableClusters * embeddingDim
        );
        numMergableClusters++;
    }

    return {numMergableClusters, subsetIndices, clusterEmbeddings};
}


vector<Cluster> AgglomerativeClustering::buildNewClusters(const vector<ClusterPair> &clusterPairs,
                                                          const vector<Cluster> *clusters) {
    auto newClusters = vector<Cluster>();
    auto numClusters = clusters->size();
    vector<bool> isClusterUsed(numClusters, false);
    vector<bool> isClusterMergable(numClusters, false);

    int numMerged = 0;
    for (const ClusterPair &clusterPair: clusterPairs) {
        isClusterMergable[clusterPair.cluster1Id] = true;
        isClusterMergable[clusterPair.cluster2Id] = true;
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
        newClusters.emplace_back(objects, embeddings, embeddingDim, true);
        numMerged++;
    }
    int numNotMerged = 0, numNonMergable = 0;
    for (int i = 0; i < numClusters; i++) {
        if (!isClusterUsed[i]) {
            newClusters.emplace_back(clusters->at(i).objects, embeddings, embeddingDim, isClusterMergable[i]);
            numNonMergable += !isClusterMergable[i];
            numNotMerged++;
        }
    }
    std::cout << "Num clusters: " << newClusters.size() << " (" << numMerged << " merged, " << numNotMerged
              << " not merged, " << numNonMergable << " non mergable)"
              << std::endl;
    return newClusters;
}

vector<Cluster> AgglomerativeClustering::mergeClusters(vector<Cluster> *clusters) {
    auto clusterEmbeddings = buildClusterEmbeddings(clusters);

    auto faissSearchResult = searchFaiss(clusterEmbeddings);

    auto clusterPairs = buildClusterPairs(clusterEmbeddings, &faissSearchResult);

    auto newClusters = buildNewClusters(clusterPairs, clusters);

    return newClusters;
}

vector<Cluster> AgglomerativeClustering::getSingletonClusters() {
    vector<Cluster> clusters;
    for (int i = 0; i < vocabSize; i++) {
        std::unordered_set<int> objects;
        objects.insert(i);
        clusters.emplace_back(objects, embeddings, embeddingDim, true);
    }
    return clusters;
}

vector<Cluster> AgglomerativeClustering::runAlgorithm() {
    auto clusters = getSingletonClusters();

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


#endif //AGGLO_CPP_AGGLOMERATIVECLUSTERING_H
