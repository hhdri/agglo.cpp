//#include <iostream>
//#include <queue>
//#include <set>
//#include <utility>
//#include <cmath>
//
//const int num_objects = 9;
//const int embedding_dim = 2;
//const double embeddings[num_objects][embedding_dim] = {
//        {1.0, 1.0},
//        {2.0, 2.0},
//        {3.0, 3.0},
//        {4.0, 4.0},
//        {5.0, 5.0},
//        {6.0, 6.0},
//        {7.0, 7.0},
//        {8.0, 8.0},
//        {90.0, 90.0}
//};
//
//class Cluster {
//public:
//    int id;
//    std::set<int> objects;
//    int parentId = -1;
//
//    Cluster(int id, std::set<int> objects) : id(id), objects(std::move(objects)) {}
//
//    friend std::ostream &operator<<(std::ostream &os, const Cluster &cluster) {
//        os << cluster.id << ", size: " << cluster.objects.size() << ", objects: ";
//        for (int object: cluster.objects) {
//            os << object << " ";
//        }
//        return os;
//    }
//};
//
//class ClusterPair {
//public:
//    int id;
//    int cluster1Id;
//    int cluster2Id;
//    double similarity;
//    std::vector<Cluster> *clusters;
//
//    explicit ClusterPair(int id, int cluster1Id, int cluster2Id, std::vector<Cluster> *clusters
//    ) : id(id), cluster1Id(cluster1Id), cluster2Id(cluster2Id), similarity(0.0), clusters(clusters) {
//        const auto &cluster1 = clusters->at(cluster1Id);
//        const auto &cluster2 = clusters->at(cluster2Id);
//        for (int object1: cluster1.objects) {
//            for (int object2: cluster2.objects) {
//                double squared_distance = 0.0;
//                for (int i = 0; i < embedding_dim; i++) {
//                    squared_distance += std::pow(embeddings[object1][i] - embeddings[object2][i], 2);
//                }
//                similarity += std::sqrt(squared_distance);
//            }
//        }
//        similarity /= (double) (cluster1.objects.size() * cluster2.objects.size());
//    }
//
//    friend std::ostream &operator<<(std::ostream &os, const ClusterPair &myObject) {
//        os << "similarity: " << myObject.similarity << " clusters: " << "\n\t" << myObject.clusters->at(myObject.cluster1Id)
//           << "\n\t" << myObject.clusters->at(myObject.cluster2Id);
//        return os;
//    }
//};
//
//struct CompareClusterPairs {
//    bool operator()(const ClusterPair &lhs, const ClusterPair &rhs) const {
//        if (lhs.similarity == rhs.similarity)
//            return lhs.id > rhs.id;
//        return lhs.similarity > rhs.similarity;
//    }
//};
//
//int main() {
//    int cluster_id = 0;
//    int clusterPairId = 0;
//
//    std::vector<Cluster> clusters;
//    std::priority_queue<ClusterPair, std::vector<ClusterPair>, CompareClusterPairs> pq;
//
//    for (; cluster_id < num_objects; cluster_id++) {
//        std::set<int> objects;
//        objects.insert(cluster_id);
//        clusters.emplace_back(cluster_id, objects);
//    }
//
//    for (int i = 0; i < clusters.size(); i++) {
//        for (int j = i + 1; j < clusters.size(); j++) {
//            pq.emplace(clusterPairId++, i, j, &clusters);
//        }
//    }
//
//    while (!pq.empty()) {
//        auto &cluster_pair = pq.top();
//        auto &clusterPair1 = clusters[cluster_pair.cluster1Id];
//        auto &clusterPair2 = clusters[cluster_pair.cluster2Id];
//        pq.pop();
//        if (clusterPair1.parentId != -1 || clusterPair2.parentId != -1) {
//            continue;
//        }
//        auto merged_set = clusterPair1.objects;
//        merged_set.insert(clusterPair2.objects.begin(), clusterPair2.objects.end());
//        clusterPair1.parentId = cluster_id;
//        clusterPair2.parentId = cluster_id;
//        clusters.emplace_back(cluster_id, merged_set);
//        for (const auto &cluster: clusters) {
//            if (cluster.id < cluster_id && cluster.parentId == -1) {
//                pq.emplace(clusterPairId++, cluster.id, cluster_id, &clusters);
//            }
//        }
//        cluster_id++;
//    }
//
//    std::cout << "Hello, World!" << std::endl;
//}
