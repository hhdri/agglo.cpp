#include <iostream>
#include <fstream>
#include <sstream>

using std::string, std::cout, std::endl;

int main() {
    const int VOCAB_SIZE = 400'000;
    const int EMBEDDING_DIM = 100;

    static float embeddings[VOCAB_SIZE][EMBEDDING_DIM];
    static string vocab[VOCAB_SIZE];

    string fileName = "/Users/majid/Downloads/glove.6B/glove.6B.100d.txt";
    std::ifstream file(fileName);

    for (int i = 0; i < VOCAB_SIZE; i++) {
        string line, word;
        std::getline(file, line);
        auto lineStream = std::istringstream(line);
        lineStream >> word;
        vocab[i] = word;

        for (int j = 0; j < EMBEDDING_DIM; j++) {
            lineStream >> embeddings[i][j];
        }
    }

    cout << vocab[VOCAB_SIZE - 1] << endl;
    cout << embeddings[VOCAB_SIZE - 1][EMBEDDING_DIM - 1] << endl;

    cout << "Done" << endl;
}