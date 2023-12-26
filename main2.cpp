#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>

#include <faiss/IndexFlat.h>

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

void
searchFaiss(float* embeddingss, float* searchDistances, faiss::idx_t* searchIndices, int embeddingDim, int vocabSize,
            int searchSize, int k) {
    faiss::IndexFlatIP index(embeddingDim);
    index.add(vocabSize, embeddingss);
    index.search(searchSize, embeddingss, k, searchDistances, searchIndices);
}

int main() {
    loadPartition({"/Users/majid/Downloads/glove.6B/glove.6B.50d.txt", 400'000}, 0);

    auto start = std::chrono::high_resolution_clock::now();

    const int SEARCH_SIZE = 400'000;
    const int K = 2;
    auto *searchIndices = new faiss::idx_t[SEARCH_SIZE * K];
    auto *searchDistances = new float[SEARCH_SIZE * K];

    searchFaiss(embeddings, searchDistances, searchIndices, EMBEDDING_DIM, VOCAB_SIZE, SEARCH_SIZE, K);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    cout << "Duration: " << duration << "ms" << endl;

    // Duration: 14618ms

//    for (int i = 0; i < SEARCH_SIZE; i++) {
//        cout << vocab[i] << endl;
//        for (int j = 0; j < K; j++) {
//            cout << "\t" << vocab[searchIndices[i * K + j]] << " " << searchDistances[i * K + j] << endl;
//        }
//    }

    cout << "Done" << endl;
}