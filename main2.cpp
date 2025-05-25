#include <iostream>
#include <fstream>
#include <sstream>
#include <queue>
#include <numeric>

using std::endl, std::cout;

constexpr int EMBEDDING_DIM = 50;
constexpr int VOCAB_SIZE = 10'000;

using IndexPair = std::pair<int, int>;
using SimilarityPair = std::pair<float, IndexPair>;

float *embeddings;
std::string *vocab;

std::string SimilarityPairString(const std::optional<SimilarityPair> &pair) {
    if (!pair.has_value()) {
        return "No similarity pair";
    }
    return std::to_string(pair.value().first) + " (" + vocab[pair.value().second.first] + ", " + vocab[pair.value().second.second] + ")";
}

int main() {
    embeddings = new float[VOCAB_SIZE * EMBEDDING_DIM];
    vocab = new std::string[VOCAB_SIZE];

    std::ifstream file("../glove.6B.50d.txt");
    for (int i = 0; i < VOCAB_SIZE; i++) {
        std::string line, word;
        std::getline(file, line);
        auto lineStream = std::istringstream(line);
        lineStream >> word;
        vocab[i] = word;

        double normSquared = 0.0;
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            float value;
            lineStream >> value;
            embeddings[i * EMBEDDING_DIM + j] = value;
            normSquared += value * value;
        }
        auto norm = std::sqrt(normSquared);
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            embeddings[i * EMBEDDING_DIM + j] /= norm;
        }
    }
    file.close();

    std::vector<std::optional<SimilarityPair>> topSims(VOCAB_SIZE);

    float negDotProduct;
    std::optional<SimilarityPair> topPair;
    for (int i = 0; i < VOCAB_SIZE; i++) {
        topPair = std::nullopt;
        for (int j = 0; j < VOCAB_SIZE; j++) {
            auto negDotProduct = -std::inner_product(
                embeddings + i * EMBEDDING_DIM,
                embeddings + (i + 1) * EMBEDDING_DIM,
                embeddings + j * EMBEDDING_DIM,
                0.0f
            );
            if (i != j && (!topPair.has_value() || negDotProduct < topPair->first)) 
                topPair = SimilarityPair(negDotProduct, IndexPair(std::min(i, j), std::max(i, j)));
        }
        if (!topSims[i].has_value() || topPair < topSims[i])
            topSims[i] = topPair;
    }
    for(int idx1 = 0; idx1 < VOCAB_SIZE; idx1++) {
        if (!topSims[idx1].has_value()) {
            continue;
        }
        auto &pair1 = topSims[idx1].value();

        auto idx2 = pair1.second.first;
        if (idx2 == idx1) {
            idx2 = pair1.second.second;
        }
        if (!topSims[idx2].has_value()) {
            continue;
        }
        auto &pair2 = topSims[idx2].value();

        auto is_equal = pair1.first == pair2.first && 
                        pair1.second.first == pair2.second.first && 
                        pair1.second.second == pair2.second.second;

        if (!is_equal) {
            cout << "Comparing: " << SimilarityPairString(topSims[idx1]) 
                 << " with " << SimilarityPairString(topSims[idx2]) << endl;
        }
        
        if (is_equal || pair1.first <= pair2.first) {
            topSims[idx2] = std::nullopt;
        } else {
            topSims[idx1] = std::nullopt;
        }
    }
    int count = 0;
    for(int i = 0; i < VOCAB_SIZE; i++) {
        if (topSims[i].has_value()) count++;
    }
    cout << "Total words with most similar: " << count << endl;
}