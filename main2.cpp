#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <set>
#include <algorithm>

#include <faiss/IndexFlat.h>

using std::string, std::cout, std::endl;

const int EMBEDDING_DIM = 50;
const int VOCAB_SIZE = 20'000;
static float embeddings[VOCAB_SIZE * EMBEDDING_DIM];
static string vocab[VOCAB_SIZE];

struct Partition {
    string path;
    int size;
};

class Cluster {
public:
    std::set<int> objects;
    float embedding[EMBEDDING_DIM];

    Cluster(std::set<int> objects) : objects(std::move(objects)) {
        auto clusterSize = (float) this->objects.size();
        for (float &i: embedding) {
            i = 0.0;
        }
        for (int object: this->objects) {
            for (int i = 0; i < EMBEDDING_DIM; i++) {
                embedding[i] += embeddings[object * EMBEDDING_DIM + i] / clusterSize;
            }
        }
    }
};

class ClusterPair {
public:
    int cluster1Id;
    int cluster2Id;
    double similarity;
};

void loadPartition(const Partition &partition, int offset) {
    std::ifstream file(partition.path);
    for (int i = offset; i < offset + partition.size; i++) {
        string line, word;
        std::getline(file, line);
        auto lineStream = std::istringstream(line);
        lineStream >> word;
        vocab[i] = word;

        float normSquared = 0.0;
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            float value;
            lineStream >> value;
            embeddings[i * EMBEDDING_DIM + j] = value;
            normSquared += value * value;
        }
        float norm = sqrt(normSquared);
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            embeddings[i * EMBEDDING_DIM + j] /= norm;
        }
    }
    file.close();
}

void
searchFaiss(float *clusterEmbeddings, float *searchDistances, faiss::idx_t *searchIndices, faiss::idx_t vocabSize,
            int searchK) {
    faiss::IndexFlatIP index(EMBEDDING_DIM);
    index.add(vocabSize, clusterEmbeddings);
    index.search(vocabSize, clusterEmbeddings, searchK, searchDistances, searchIndices);
}

void buildClusterPairs(unsigned long numClusters, int searchK, const float *searchDistances,
                       const faiss::idx_t *searchIndices,
                       std::vector<ClusterPair> &clusterPairs) {
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
buildClusterEmbeddings(const std::vector<Cluster> *clusters, float *clusterEmbeddings) {
    auto numClusters = clusters->size();
    for (int i = 0; i < numClusters; i++) {
        std::copy(clusters->at(i).embedding, clusters->at(i).embedding + EMBEDDING_DIM,
                  clusterEmbeddings + i * EMBEDDING_DIM);
    }
}

void populateNewClusters(const std::vector<ClusterPair> &clusterPairs, const std::vector<Cluster> *clusters,
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
        std::set<int> objects;
        objects.insert(clusters->at(clusterPair.cluster1Id).objects.begin(),
                       clusters->at(clusterPair.cluster1Id).objects.end());
        objects.insert(clusters->at(clusterPair.cluster2Id).objects.begin(),
                       clusters->at(clusterPair.cluster2Id).objects.end());
        newClusters->emplace_back(objects);
        numMerged++;
    }
    int numNotMerged = 0;
    for (int i = 0; i < numClusters; i++) {
        if (!isClusterUsed[i]) {
            newClusters->emplace_back(clusters->at(i).objects);
            numNotMerged++;
        }
    }
    cout << "Num clusters: " << newClusters->size() << " (" << numMerged << " merged, " << numNotMerged
         << " not merged)"
         << endl;
}

void mergeClusters(std::vector<Cluster> *clusters, std::vector<Cluster> *newClusters, int searchK) {
    unsigned long num_clusters = clusters->size();
    auto *clusterEmbeddings = new float[num_clusters * EMBEDDING_DIM];
    buildClusterEmbeddings(clusters, clusterEmbeddings);

    auto *searchIndices = new faiss::idx_t[num_clusters * searchK];
    auto *searchDistances = new float[num_clusters * searchK];
    searchFaiss(clusterEmbeddings, searchDistances, searchIndices, (faiss::idx_t) num_clusters, searchK);

    delete[] clusterEmbeddings;

    std::vector<ClusterPair> clusterPairs;
    buildClusterPairs(num_clusters, searchK, searchDistances, searchIndices, clusterPairs);

    delete[] searchIndices;
    delete[] searchDistances;

    populateNewClusters(clusterPairs, clusters, newClusters);
}

void buildSingletonClusters(std::vector<Cluster> *clusters) {
    for (int i = 0; i < VOCAB_SIZE; i++) {
        std::set<int> objects;
        objects.insert(i);
        clusters->emplace_back(objects);
    }
}

int main() {
    const int searchK = 10;

    loadPartition({"/Users/majid/Downloads/glove.6B/glove.6B.50d/glove.6B.50d.aa", 20'000}, 0);

    std::vector<Cluster> clusters, newClusters;
    buildSingletonClusters(&clusters);

    for (int i = 0;; i++) {
        cout << "Iteration " << i << endl;
        mergeClusters(&clusters, &newClusters, searchK);
        if (newClusters.size() == clusters.size()) {
            break;
        }
        clusters = newClusters;
        newClusters.clear();
    }

    // Sort clusters based on size
    std::sort(newClusters.begin(), newClusters.end(),
              [](const Cluster &a, const Cluster &b) { return a.objects.size() > b.objects.size(); });

    std::vector<std::vector<string>> clusterObjects;

    for (const Cluster &cluster: newClusters) {
        std::vector<string> objects;
        objects.reserve(cluster.objects.size());
        for (int object: cluster.objects) {
            objects.push_back(vocab[object]);
        }
        clusterObjects.push_back(objects);
    }

    cout << "Done" << endl;
}