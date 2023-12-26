#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <vector>
#include <queue>
#include <numeric>
#include <algorithm>
#include <list>

using std::string, std::cout, std::endl;

const int VOCAB_SIZE = 400'000;
const int EMBEDDING_DIM = 50;
static float embeddings[VOCAB_SIZE * EMBEDDING_DIM];
static string vocab[VOCAB_SIZE];

struct Partition {
    string path;
    int size;
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

int main() {
    loadPartition({"/Users/majid/Downloads/glove.6B/glove.6B.50d.txt", 400'000}, 0);

    auto start = std::chrono::high_resolution_clock::now();

    const int SEARCH_SIZE = 500;
    const int K = 5;

    auto *searchIndices = new long long[SEARCH_SIZE * K];
    auto *searchDistances = new float[SEARCH_SIZE * K];

    for (int i = 0; i < SEARCH_SIZE; i++) {
        float InnerSearchSimilarities[VOCAB_SIZE];
        long long innerSearchIndices[VOCAB_SIZE];
        std::iota(std::begin(innerSearchIndices), std::end(innerSearchIndices), 0);
        for (int j = 0; j < VOCAB_SIZE; j++) {
            InnerSearchSimilarities[j] = 0.0;
            for (int k = 0; k < EMBEDDING_DIM; k++) {
                InnerSearchSimilarities[j] += embeddings[i * EMBEDDING_DIM + k] * embeddings[j * EMBEDDING_DIM + k];
            }
        }
        std::sort(std::begin(innerSearchIndices), std::end(innerSearchIndices),
                  [&](int i, int j) { return InnerSearchSimilarities[i] > InnerSearchSimilarities[j]; });
        for (int j = 0; j < K; j++) {
            searchIndices[i * K + j] = innerSearchIndices[j];
            searchDistances[i * K + j] = InnerSearchSimilarities[innerSearchIndices[j]];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    cout << "Duration: " << duration << "ms" << endl;

    // Duration: 70613ms
    // Duration: 58414ms

    for (int i = 0; i < SEARCH_SIZE; i++) {
        cout << vocab[i] << endl;
        for (int j = 0; j < 5; j++) {
            cout << "\t" << vocab[searchIndices[i * 5 + j]] << " " << searchDistances[i * 5 + j] << endl;
        }
    }

    cout << "Done" << endl;
}