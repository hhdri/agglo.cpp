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

int main() {
    loadPartition({"/Users/majid/Downloads/glove.6B/glove.6B.50d.txt", 400'000}, 0);

    auto start = std::chrono::high_resolution_clock::now();

    faiss::IndexFlatIP index(EMBEDDING_DIM);
    index.add(VOCAB_SIZE, embeddings);

    const int SEARCH_SIZE = 50'000;

    auto *searchIndices = new long long[SEARCH_SIZE * 5];
    auto *searchDistances = new float[SEARCH_SIZE * 5];

    index.search(SEARCH_SIZE, embeddings, 5, searchDistances, searchIndices);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    cout << "Duration: " << duration << "ms" << endl;

    // Duration: 14618ms

//    for (int i = 0; i < SEARCH_SIZE; i++) {
//        cout << vocab[i] << endl;
//        for (int j = 0; j < 5; j++) {
//            cout << "\t" << vocab[searchIndices[i * 5 + j]] << " " << searchDistances[i * 5 + j] << endl;
//        }
//    }

    cout << "Done" << endl;
}