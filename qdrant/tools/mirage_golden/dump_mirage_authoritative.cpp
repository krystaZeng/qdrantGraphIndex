// Authoritative MIRAGE Layer-0 injection fixture dumper.
//
// Unlike dump_mirage_golden.cpp, this file is intended to be built against the
// local FAISS/MIRAGE C++ reference implementation. The critical field
// `layer0_injected` is dumped from the real hierarchy.hnsw.neighbors table
// after calling IndexHNSW::init_level_0_from_knngraph(k = S, D, I).
//
// This source is intentionally small and avoids duplicating the HNSW shrink
// heuristic. Build it from an environment where the local FAISS sources can be
// compiled with their normal OpenMP/BLAS dependencies.

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/MIRAGE.h>

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {

constexpr int kN = 64;
constexpr int kDim = 8;
constexpr int kS = 16;
constexpr int kR = 4;
constexpr int kIter = 15;
constexpr int kNumReverseEdges = 96;
constexpr int kMirageSeed = 2021;
constexpr int kHnswM = 16;
constexpr int kHnswM0 = 2 * kHnswM;
constexpr int kLayer0K = kS;
constexpr int kHnswLevelSeed = 12345;
constexpr int kShuffleSeed = 789;

struct Layer0Stats {
    int total_edges = 0;
    int min_degree = 0;
    int max_degree = 0;
    double avg_degree = 0.0;
    std::map<int, int> degree_histogram;
    int reciprocal_edges = 0;
    double reciprocity_ratio = 0.0;
    int weak_connected_components = 0;
    int largest_weak_component = 0;
};

float vector_value(int i, int dim) {
    const int raw = ((i + 1) * 37 + (dim + 3) * 17 + (i * dim) * 13) % 1009;
    return (static_cast<float>(raw) - 504.0f) / 100.0f;
}

std::vector<float> generate_flat_vectors() {
    std::vector<float> vectors(kN * kDim);
    for (int i = 0; i < kN; ++i) {
        for (int d = 0; d < kDim; ++d) {
            vectors[i * kDim + d] = vector_value(i, d);
        }
    }
    return vectors;
}

std::vector<int> take_top_s(const faiss::Mirage& mirage, int u) {
    const int offset = mirage.offsets[u];
    const int degree = mirage.offsets[u + 1] - offset;
    const int take = std::min(kLayer0K, degree);

    std::vector<int> result;
    result.reserve(take);
    for (int i = 0; i < take; ++i) {
        result.push_back(mirage.final_graph[offset + i]);
    }
    return result;
}

std::vector<std::vector<int>> dump_mirage_layer0(const faiss::Mirage& mirage) {
    std::vector<std::vector<int>> result(kN);
    for (int u = 0; u < kN; ++u) {
        const int offset = mirage.offsets[u];
        const int degree = mirage.offsets[u + 1] - offset;
        result[u].reserve(degree);
        for (int i = 0; i < degree; ++i) {
            result[u].push_back(mirage.final_graph[offset + i]);
        }
    }
    return result;
}

std::vector<std::vector<int>> dump_layer0_top_s(const faiss::Mirage& mirage) {
    std::vector<std::vector<int>> result(kN);
    for (int u = 0; u < kN; ++u) {
        result[u] = take_top_s(mirage, u);
    }
    return result;
}

std::vector<std::vector<int>> dump_injected_layer0(const faiss::IndexHNSW& hierarchy) {
    std::vector<std::vector<int>> result(kN);
    for (int u = 0; u < kN; ++u) {
        size_t begin = 0;
        size_t end = 0;
        hierarchy.hnsw.neighbor_range(u, 0, &begin, &end);

        for (size_t pos = begin; pos < end; ++pos) {
            const int neighbor = hierarchy.hnsw.neighbors[pos];
            if (neighbor < 0) {
                continue;
            }
            result[u].push_back(neighbor);
        }
    }
    return result;
}

std::vector<int> generate_levels() {
    faiss::HNSW hnsw(kHnswM);
    hnsw.rng = faiss::RandomGenerator(kHnswLevelSeed);

    std::vector<int> levels;
    levels.reserve(kN);
    for (int i = 0; i < kN; ++i) {
        levels.push_back(hnsw.random_level());
    }
    return levels;
}

std::map<int, std::vector<int>> upper_insert_order_by_level(const std::vector<int>& levels) {
    int max_level = 0;
    for (int level : levels) {
        max_level = std::max(max_level, level);
    }

    std::vector<std::vector<int>> buckets(max_level + 1);
    for (int i = 0; i < static_cast<int>(levels.size()); ++i) {
        buckets[levels[i]].push_back(i);
    }

    faiss::RandomGenerator rng(kShuffleSeed);
    std::map<int, std::vector<int>> order;
    for (int level = max_level; level > 0; --level) {
        auto& bucket = buckets[level];
        for (int j = 0; j < static_cast<int>(bucket.size()); ++j) {
            std::swap(bucket[j], bucket[j + rng.rand_int(static_cast<int>(bucket.size()) - j)]);
        }
        order[level] = bucket;
    }
    return order;
}

Layer0Stats compute_layer0_stats(const std::vector<std::vector<int>>& graph) {
    Layer0Stats stats;
    if (graph.empty()) {
        return stats;
    }

    stats.min_degree = static_cast<int>(graph[0].size());
    std::set<std::pair<int, int>> edges;

    for (int u = 0; u < static_cast<int>(graph.size()); ++u) {
        const int degree = static_cast<int>(graph[u].size());
        stats.total_edges += degree;
        stats.min_degree = std::min(stats.min_degree, degree);
        stats.max_degree = std::max(stats.max_degree, degree);
        stats.degree_histogram[degree] += 1;
        for (int v : graph[u]) {
            edges.emplace(u, v);
        }
    }

    stats.avg_degree = static_cast<double>(stats.total_edges) /
            static_cast<double>(graph.size());

    for (const auto& [u, v] : edges) {
        if (edges.find({v, u}) != edges.end()) {
            stats.reciprocal_edges += 1;
        }
    }
    stats.reciprocity_ratio = stats.total_edges == 0
            ? 0.0
            : static_cast<double>(stats.reciprocal_edges) /
                    static_cast<double>(stats.total_edges);

    std::vector<std::vector<int>> undirected(graph.size());
    for (const auto& [u, v] : edges) {
        if (v < 0 || v >= static_cast<int>(graph.size())) {
            continue;
        }
        undirected[u].push_back(v);
        undirected[v].push_back(u);
    }

    std::vector<bool> visited(graph.size(), false);
    std::vector<int> stack;
    for (int start = 0; start < static_cast<int>(graph.size()); ++start) {
        if (visited[start]) {
            continue;
        }
        stats.weak_connected_components += 1;
        int component_size = 0;
        stack.clear();
        stack.push_back(start);
        visited[start] = true;

        while (!stack.empty()) {
            const int u = stack.back();
            stack.pop_back();
            component_size += 1;

            for (int v : undirected[u]) {
                if (!visited[v]) {
                    visited[v] = true;
                    stack.push_back(v);
                }
            }
        }

        stats.largest_weak_component =
                std::max(stats.largest_weak_component, component_size);
    }

    return stats;
}

void print_vec(const std::vector<int>& values) {
    std::cout << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            std::cout << ", ";
        }
        std::cout << values[i];
    }
    std::cout << "]";
}

void print_nested_vec(const std::vector<std::vector<int>>& values) {
    std::cout << "[\n";
    for (size_t i = 0; i < values.size(); ++i) {
        std::cout << "    ";
        print_vec(values[i]);
        if (i + 1 != values.size()) {
            std::cout << ",";
        }
        std::cout << "\n";
    }
    std::cout << "  ]";
}

void print_int_map(const std::map<int, int>& values) {
    std::cout << "{";
    for (auto it = values.begin(); it != values.end(); ++it) {
        if (it != values.begin()) {
            std::cout << ", ";
        }
        std::cout << "\"" << it->first << "\": " << it->second;
    }
    std::cout << "}";
}

void print_stats(const Layer0Stats& stats) {
    std::cout << "{\n";
    std::cout << "    \"total_edges\": " << stats.total_edges << ",\n";
    std::cout << "    \"min_degree\": " << stats.min_degree << ",\n";
    std::cout << "    \"max_degree\": " << stats.max_degree << ",\n";
    std::cout << "    \"avg_degree\": " << std::fixed << std::setprecision(6)
              << stats.avg_degree << ",\n";
    std::cout << "    \"degree_histogram\": ";
    print_int_map(stats.degree_histogram);
    std::cout << ",\n";
    std::cout << "    \"reciprocal_edges\": " << stats.reciprocal_edges << ",\n";
    std::cout << "    \"reciprocity_ratio\": " << std::fixed << std::setprecision(6)
              << stats.reciprocity_ratio << ",\n";
    std::cout << "    \"weak_connected_components\": "
              << stats.weak_connected_components << ",\n";
    std::cout << "    \"largest_weak_component\": "
              << stats.largest_weak_component << "\n";
    std::cout << "  }";
}

} // namespace

int main() {
    omp_set_num_threads(1);

    const auto flat_vectors = generate_flat_vectors();

    faiss::IndexFlatL2 storage(kDim);
    storage.add(kN, flat_vectors.data());

    std::unique_ptr<faiss::DistanceComputer> distance_computer(
            storage.get_distance_computer());

    faiss::Mirage mirage(kDim);
    mirage.S = kS;
    mirage.R = kR;
    mirage.iter = kIter;
    mirage.random_seed = kMirageSeed;
    mirage.build(*distance_computer, kN, false);

    faiss::IndexHNSW hierarchy(&storage, kHnswM);
    hierarchy.ntotal = kN;
    hierarchy.d = kDim;
    hierarchy.hnsw.prepare_level_tab_mirage(kN, false);

    std::vector<float> distances(kN * kLayer0K, 0.0f);
    std::vector<faiss::idx_t> labels(kN * kLayer0K, -1);

    for (int u = 0; u < kN; ++u) {
        std::vector<float> query(kDim);
        storage.reconstruct(u, query.data());
        std::unique_ptr<faiss::DistanceComputer> local_distance_computer(
                storage.get_distance_computer());
        local_distance_computer->set_query(query.data());

        const int offset = mirage.offsets[u];
        const int degree = mirage.offsets[u + 1] - offset;
        for (int j = 0; j < degree && j < kLayer0K; ++j) {
            const int neighbor = mirage.final_graph[offset + j];
            labels[u * kLayer0K + j] = neighbor;
            distances[u * kLayer0K + j] = (*local_distance_computer)(neighbor);
        }
    }

    hierarchy.init_level_0_from_knngraph(
            kLayer0K, distances.data(), labels.data());

    const auto levels = generate_levels();
    const auto upper_order = upper_insert_order_by_level(levels);
    const auto layer0 = dump_mirage_layer0(mirage);
    const auto layer0_top_s = dump_layer0_top_s(mirage);
    const auto layer0_injected = dump_injected_layer0(hierarchy);
    const auto layer0_injected_stats = compute_layer0_stats(layer0_injected);

    std::cout << "{\n";
    std::cout << "  \"version\": 1,\n";
    std::cout << "  \"source\": \"tools/mirage_golden/dump_mirage_authoritative.cpp uses real C++ Mirage::build and IndexHNSW::init_level_0_from_knngraph\",\n";
    std::cout << "  \"metric\": \"l2\",\n";
    std::cout << "  \"vector_generator\": \"integer-affine-v1\",\n";
    std::cout << "  \"layer0_injected_source\": \"authoritative_cpp_hierarchy_hnsw_neighbors\",\n";
    std::cout << "  \"n\": " << kN << ",\n";
    std::cout << "  \"dim\": " << kDim << ",\n";
    std::cout << "  \"params\": {\n";
    std::cout << "    \"s\": " << kS << ",\n";
    std::cout << "    \"r\": " << kR << ",\n";
    std::cout << "    \"iter\": " << kIter << ",\n";
    std::cout << "    \"num_reverse_edges\": " << kNumReverseEdges << ",\n";
    std::cout << "    \"mirage_seed\": " << kMirageSeed << ",\n";
    std::cout << "    \"hnsw_m\": " << kHnswM << ",\n";
    std::cout << "    \"hnsw_m0\": " << kHnswM0 << ",\n";
    std::cout << "    \"layer0_k\": " << kLayer0K << ",\n";
    std::cout << "    \"hnsw_level_seed\": " << kHnswLevelSeed << ",\n";
    std::cout << "    \"shuffle_seed\": " << kShuffleSeed << ",\n";
    std::cout << "    \"threads\": 1\n";
    std::cout << "  },\n";
    std::cout << "  \"levels_zero_based\": ";
    print_vec(levels);
    std::cout << ",\n";
    std::cout << "  \"upper_insert_order_by_level\": {\n";
    for (auto it = upper_order.begin(); it != upper_order.end(); ++it) {
        std::cout << "    \"" << it->first << "\": ";
        print_vec(it->second);
        if (std::next(it) != upper_order.end()) {
            std::cout << ",";
        }
        std::cout << "\n";
    }
    std::cout << "  },\n";
    std::cout << "  \"layer0\": ";
    print_nested_vec(layer0);
    std::cout << ",\n";
    std::cout << "  \"layer0_top_s\": ";
    print_nested_vec(layer0_top_s);
    std::cout << ",\n";
    std::cout << "  \"layer0_injected\": ";
    print_nested_vec(layer0_injected);
    std::cout << ",\n";
    std::cout << "  \"layer0_injected_stats\": ";
    print_stats(layer0_injected_stats);
    std::cout << "\n";
    std::cout << "}\n";
}
