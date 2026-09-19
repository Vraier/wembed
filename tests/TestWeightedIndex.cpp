#include <gtest/gtest.h>

#include <algorithm>
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

}  // namespace

// The dynamic-query cache (rebuilds amortized under a movement budget) must return
// exactly the same repelling pairs as an index that is rebuilt every step. The random
// walk is sized so the run passes through fill, several reuse steps and forced rebuilds.
TEST(TestWeightedIndex, DynamicReuseMatchesFreshRebuild) {
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

    WeightedIndex dynamic(IndexType::Sprk, DIM, 4.0, 0.5);
    WeightedIndex fresh(IndexType::Sprk, DIM, 4.0, 0.0);

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
