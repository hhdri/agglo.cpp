#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>

using std::string, std::cout, std::endl;

const int VOCAB_SIZE = 400'000;
const int EMBEDDING_DIM = 300;
static float embeddings[VOCAB_SIZE][EMBEDDING_DIM];
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

        for (int j = 0; j < EMBEDDING_DIM; j++) {
            lineStream >> embeddings[i][j];
        }
    }

    file.close();
}

int main() {
    const string pathPrefix = "/Users/majid/Downloads/glove.6B/glove.6B.300d/glove.6B.300d.";
    const Partition partitions[] = {
            {pathPrefix + "aa", 50'000},
            {pathPrefix + "ab", 50'000},
            {pathPrefix + "ac", 50'000},
            {pathPrefix + "ad", 50'000},
            {pathPrefix + "ae", 50'000},
            {pathPrefix + "af", 50'000},
            {pathPrefix + "ag", 50'000},
            {pathPrefix + "ah", 50'000},
    };

//    std::thread threads[8];
//
//    for (int i = 0; i < 8; i++) {
//        threads[i] = std::thread(loadPartition, partitions[i], i * 50'000);
//    }
//
//    for (auto &thread: threads) {
//        thread.join();
//    }

//    loadPartition(partitions[0], 0);
//    loadPartition(partitions[1], 50'000);
//    loadPartition(partitions[2], 100'000);
//    loadPartition(partitions[3], 150'000);
//    loadPartition(partitions[4], 200'000);
//    loadPartition(partitions[5], 250'000);
//    loadPartition(partitions[6], 300'000);
//    loadPartition(partitions[7], 350'000);

    loadPartition({"/Users/majid/Downloads/glove.6B/glove.6B.300d.txt", 400'000}, 0);

    cout << vocab[0] << endl;
    cout << embeddings[0][EMBEDDING_DIM - 1] << endl;

    int idx = 359685;
    cout << vocab[idx] << endl;
    cout << embeddings[idx][2] << endl;

    cout << "Done" << endl;
}