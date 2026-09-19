#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "WeightedIndex.hpp"

namespace {

constexpr int DIM = 2;
constexpr int N = 300;

std::vector<NodeId> sortedPairs(WeightedIndex& index, NodeId v) {
    std::vector<NodeId> out;
    index.getOwnedRepellingPairs(v, out);
    std::sort(out.begin(), out.end());
    return out;
}

// the sprk tree only exists if wembed was built with Rust
std::vector<IndexType> availableIndexTypes() {
    std::vector<IndexType> types = {IndexType::KdTree};
#ifdef WEMBED_HAS_SPRK
    types.push_back(IndexType::Sprk);
#endif
    return types;
}

}  // namespace

class TestWeightedIndex : public testing::TestWithParam<IndexType> {};

INSTANTIATE_TEST_SUITE_P(IndexTypes, TestWeightedIndex, testing::ValuesIn(availableIndexTypes()),
                         [](const testing::TestParamInfo<IndexType>& info) { return indexTypeMap.at(info.param); });

// Nobody should end up with the slower KD-tree, or without repulsion, by accident
TEST(TestWeightedIndexDeathTest, UnusableSprkTreeIsAnError) {
    EXPECT_DEATH(WeightedIndex(IndexType::Sprk, 17, 4.0, 0.0), "");
    EXPECT_DEATH(WeightedIndex(IndexType::Sprk, 1, 4.0, 0.0), "");
#ifndef WEMBED_HAS_SPRK
    EXPECT_DEATH(WeightedIndex(IndexType::Sprk, DIM, 4.0, 0.0), "");
#endif
}

// The owned pairs are defined without any reference to the index: v owns {v, u} if it is
// the heavier endpoint, and the pair repels if its weighted distance is below 1.
TEST_P(TestWeightedIndex, OwnedPairsMatchBruteForce) {
    std::mt19937 gen(17);
    for (const int dim : {2, 4, 8, 20}) {
        if (GetParam() == IndexType::Sprk && dim > 16) continue;  // more than the sprk tree supports
        // keeps the expected number of repelling pairs per node roughly independent of dim
        std::uniform_real_distribution<double> posDist(0.0, std::pow(2.0 * N, 1.0 / dim) * (dim > 8 ? 0.6 : 1.0));
        std::uniform_real_distribution<double> weightExponent(0.0, 3.0);  // weights over 3 orders of magnitude

        VecList positions(dim, N);
        std::vector<double> weights(N);
        for (int v = 0; v < N; v++) {
            for (int d = 0; d < dim; d++) positions[v][d] = posDist(gen);
            weights[v] = std::pow(10.0, weightExponent(gen));
        }
        weights[1] = weights[0];  // tie, owned by the smaller id

        WeightedIndex index(GetParam(), dim, 4.0, 0.0);
        index.update(positions, weights, std::numeric_limits<double>::infinity());

        size_t numPairs = 0;
        for (NodeId v = 0; v < N; v++) {
            std::vector<NodeId> expected;
            for (NodeId u = 0; u < N; u++) {
                if (u == v || !WeightedIndex::ownsPair(weights[v], weights[u], v, u)) continue;
                double distSq = 0.0;
                for (int d = 0; d < dim; d++) {
                    distSq += (positions[v][d] - positions[u][d]) * (positions[v][d] - positions[u][d]);
                }
                if (std::sqrt(distSq) / std::pow(weights[v] * weights[u], 1.0 / dim) < 1.0) expected.push_back(u);
            }
            numPairs += expected.size();
            EXPECT_EQ(sortedPairs(index, v), expected) << "node " << v << " dim " << dim;
        }
        EXPECT_GT(numPairs, N) << "test instance is too sparse to be meaningful";
    }
}

// The dynamic-query cache (rebuilds amortized under a movement budget) must return
// exactly the same repelling pairs as an index that is rebuilt every step. The random
// walk is sized so the run passes through fill, several reuse steps and forced rebuilds.
TEST_P(TestWeightedIndex, DynamicReuseMatchesFreshRebuild) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> posDist(0.0, 15.0);
    std::uniform_real_distribution<double> weightDist(1.0, 10.0);
    std::uniform_real_distribution<double> moveDist(-0.05, 0.05);

    VecList positions(DIM, N);
    std::vector<double> weights(N);
    for (int v = 0; v < N; v++) {
        for (int d = 0; d < DIM; d++) positions[v][d] = posDist(gen);
        weights[v] = weightDist(gen);
    }

    WeightedIndex dynamic(GetParam(), DIM, 4.0, 0.5);
    WeightedIndex fresh(GetParam(), DIM, 4.0, 0.0);

    const double inf = std::numeric_limits<double>::infinity();
    double maxDisplacement = inf;
    for (int step = 0; step < 25; step++) {
        dynamic.update(positions, weights, maxDisplacement);
        fresh.update(positions, weights, inf);
        for (NodeId v = 0; v < N; v++) {
            EXPECT_EQ(sortedPairs(dynamic, v), sortedPairs(fresh, v)) << "node " << v << " step " << step;
        }

        maxDisplacement = 0.0;
        for (int v = 0; v < N; v++) {
            double distSq = 0.0;
            for (int d = 0; d < DIM; d++) {
                const double delta = moveDist(gen);
                positions[v][d] += delta;
                distSq += delta * delta;
            }
            maxDisplacement = std::max(maxDisplacement, std::sqrt(distSq));
        }
    }
    EXPECT_GT(dynamic.numRebuilds(), 1);
    EXPECT_LT(dynamic.numRebuilds(), dynamic.numUpdates());
}
