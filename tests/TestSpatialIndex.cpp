#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "KdTreeQueries.hpp"
#include "SpacialIndex.hpp"
#include "VecList.hpp"
#ifdef WEMBED_HAS_SPRK
#include "SprkQueries.hpp"
#endif

// Tests of the SpatialIndex interface. They run for every implementation that is listed
// in spatialIndices(); a new index only has to be added there.

namespace {

struct IndexUnderTest {
    std::string name;
    int minDimension;
    int maxDimension;
    std::function<std::unique_ptr<SpatialIndex>(const std::vector<CVecRef>&, size_t)> build;
};

void PrintTo(const IndexUnderTest& index, std::ostream* os) { *os << index.name; }

std::vector<IndexUnderTest> spatialIndices() {
    std::vector<IndexUnderTest> indices;
    indices.push_back({"KdTree", 1, 64, [](const std::vector<CVecRef>& points, size_t dimension) {
                           return std::make_unique<KdTreeQueries>(points, dimension);
                       }});
#ifdef WEMBED_HAS_SPRK
    indices.push_back({"Sprk", 2, 16, [](const std::vector<CVecRef>& points, size_t dimension) {
                           return std::make_unique<SprkQueries>(points, dimension);
                       }});
#endif
    return indices;
}

std::vector<CVecRef> refs(const VecList& positions) {
    std::vector<CVecRef> result;
    for (int i = 0; i < positions.size(); i++) result.push_back(positions[i]);
    return result;
}

double sqDistance(CVecRef a, CVecRef b) {
    double sum = 0.0;
    for (int k = 0; k < a.dimension(); k++) sum += (a[k] - b[k]) * (a[k] - b[k]);
    return sum;
}

// The indices work in float32, which blurs the boundary of the query ball: points that are
// clearly inside must be reported, points that are clearly outside must not, and the ids
// must be valid and unique. The tolerance covers the rounding of squared distances, which
// grows with the magnitude of the coordinates.
void expectMatchesBruteForce(const SpatialIndex& index, const VecList& positions, CVecRef query, double radius) {
    std::vector<uint64_t> found = {12345};  // stale content has to be cleared by the query
    const size_t numFound = index.query_sphere(query, radius, found);
    ASSERT_EQ(numFound, found.size());
    std::sort(found.begin(), found.end());
    ASSERT_TRUE(std::adjacent_find(found.begin(), found.end()) == found.end()) << "duplicate ids";
    for (const uint64_t id : found) ASSERT_LT(id, positions.size());

    double maxAbs = 0.0;
    for (int k = 0; k < query.dimension(); k++) maxAbs = std::max(maxAbs, std::abs(query[k]));
    for (int i = 0; i < positions.size(); i++) {
        for (int k = 0; k < positions.dimension(); k++) maxAbs = std::max(maxAbs, std::abs(positions[i][k]));
    }
    const double sqTolerance = 1e-3 + 1e-5 * positions.dimension() * maxAbs * maxAbs;

    for (int i = 0; i < positions.size(); i++) {
        const double sqDist = sqDistance(positions[i], query);
        const bool reported = std::binary_search(found.begin(), found.end(), static_cast<uint64_t>(i));
        if (sqDist <= radius * radius - sqTolerance) {
            EXPECT_TRUE(reported) << "missed point " << i << " at distance " << std::sqrt(sqDist);
        } else if (sqDist > radius * radius + sqTolerance) {
            EXPECT_FALSE(reported) << "reported point " << i << " at distance " << std::sqrt(sqDist);
        }
    }
}

// mixture of a uniform background and a few dense clusters, like an embedding
VecList clusteredPoints(int dimension, int n, std::mt19937& gen, double offset = 0.0) {
    std::uniform_real_distribution<double> uniform(-10.0, 10.0);
    std::normal_distribution<double> normal(0.0, 0.3);
    std::vector<std::vector<double>> centres(5, std::vector<double>(dimension));
    for (auto& centre : centres) {
        for (double& c : centre) c = uniform(gen);
    }
    VecList positions(dimension, n);
    for (int i = 0; i < n; i++) {
        const bool inCluster = i % 2 == 0;
        for (int k = 0; k < dimension; k++) {
            positions[i][k] = offset + (inCluster ? centres[i % centres.size()][k] + normal(gen) : uniform(gen));
        }
    }
    return positions;
}

}  // namespace

class TestSpatialIndex : public testing::TestWithParam<IndexUnderTest> {};

INSTANTIATE_TEST_SUITE_P(Indices, TestSpatialIndex, testing::ValuesIn(spatialIndices()),
                         [](const testing::TestParamInfo<IndexUnderTest>& info) { return info.param.name; });

TEST_P(TestSpatialIndex, EmptyIndex) {
    VecList query(3, 1);
    query.setAll(0.0);
    const auto index = GetParam().build({}, 3);
    std::vector<uint64_t> found = {42};
    EXPECT_EQ(index->query_sphere(query[0], 100.0, found), 0);
    EXPECT_TRUE(found.empty());
}

TEST_P(TestSpatialIndex, MatchesBruteForce) {
    std::mt19937 gen(1234);
    std::uniform_real_distribution<double> queryDist(-12.0, 12.0);
    for (const int dimension : {1, 2, 3, 4, 8, 16, 20}) {
        if (dimension < GetParam().minDimension || dimension > GetParam().maxDimension) continue;
        // tiny inputs, and sizes around the leaf sizes of the trees (32 and 500)
        for (const int n : {1, 2, 31, 32, 33, 65, 499, 500, 501, 1000, 3000}) {
            const VecList positions = clusteredPoints(dimension, n, gen);
            const auto index = GetParam().build(refs(positions), dimension);

            VecList query(dimension, 1);
            for (int repeat = 0; repeat < 30; repeat++) {
                if (repeat % 2 == 0) {
                    for (int k = 0; k < dimension; k++) query[0][k] = queryDist(gen);
                } else {
                    // queries usually sit on a point of some index
                    for (int k = 0; k < dimension; k++) query[0][k] = positions[repeat % n][k];
                }
                for (const double radius : {0.0, 0.2, 1.5, 6.0, 1000.0}) {
                    SCOPED_TRACE("d=" + std::to_string(dimension) + " n=" + std::to_string(n) +
                                 " r=" + std::to_string(radius));
                    expectMatchesBruteForce(*index, positions, query[0], radius);
                }
            }
        }
    }
}

// Many points share coordinates, so ties end up on both sides of a split plane
TEST_P(TestSpatialIndex, DuplicatesAndTies) {
    std::mt19937 gen(99);
    std::uniform_int_distribution<int> grid(0, 3);
    const int dimension = 3;
    const int n = 2000;

    VecList onGrid(dimension, n);
    VecList identical(dimension, n);
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < dimension; k++) {
            onGrid[i][k] = grid(gen);
            identical[i][k] = 0.5;
        }
    }

    const auto gridIndex = GetParam().build(refs(onGrid), dimension);
    const auto identicalIndex = GetParam().build(refs(identical), dimension);
    for (int i = 0; i < 50; i++) {
        // no distance between grid points is close to one of these radii
        for (const double radius : {0.5, 1.2, 1.6, 2.5}) {
            expectMatchesBruteForce(*gridIndex, onGrid, onGrid[i], radius);
            expectMatchesBruteForce(*identicalIndex, identical, onGrid[i], radius);
        }
    }

    std::vector<uint64_t> found;
    EXPECT_EQ(identicalIndex->query_sphere(identical[0], 0.0, found), n);
}

// Large enough for a parallel build; queried concurrently like the embedder does
TEST_P(TestSpatialIndex, ConcurrentQueriesAndDeterminism) {
    std::mt19937 gen(5);
    const int dimension = 4;
    const int n = 60000;
    const VecList positions = clusteredPoints(dimension, n, gen);
    const auto index = GetParam().build(refs(positions), dimension);
    const auto sameIndex = GetParam().build(refs(positions), dimension);

    const int numQueries = 400;
    std::vector<std::vector<uint64_t>> results(numQueries);
    std::vector<std::vector<uint64_t>> resultsRebuilt(numQueries);
#pragma omp parallel for schedule(dynamic, 8)
    for (int q = 0; q < numQueries; q++) {
        index->query_sphere(positions[q * 131 % n], 0.8, results[q]);
        sameIndex->query_sphere(positions[q * 131 % n], 0.8, resultsRebuilt[q]);
    }

    for (int q = 0; q < numQueries; q++) {
        // the embedder is only deterministic if the index is, including the order of the ids
        EXPECT_EQ(results[q], resultsRebuilt[q]);
    }
    for (int q = 0; q < numQueries; q += 10) {
        expectMatchesBruteForce(*index, positions, positions[q * 131 % n], 0.8);
    }
}

// Stronger than what the interface promises: the KD-tree pads its radius by the float32
// rounding error, so that it never misses a point within the radius. That has to hold for
// points exactly on the boundary, also far from the origin where float32 is coarse.
TEST(TestKdTreeQueries, BoundaryPointsAreNotMissed) {
    std::mt19937 gen(7);
    for (const double offset : {0.0, 1e3, 1e5}) {
        for (const int dimension : {2, 4, 8}) {
            const int n = 500;
            const VecList positions = clusteredPoints(dimension, n, gen, offset);
            const KdTreeQueries tree(refs(positions), dimension);
            for (int i = 0; i < n; i++) {
                const int other = (i * 7 + 1) % n;
                const double radius = std::sqrt(sqDistance(positions[i], positions[other]));
                std::vector<uint64_t> found;
                tree.query_sphere(positions[i], radius, found);
                EXPECT_TRUE(std::find(found.begin(), found.end(), other) != found.end())
                    << "offset " << offset << " d=" << dimension << " pair " << i << " " << other;
            }
        }
    }
}
