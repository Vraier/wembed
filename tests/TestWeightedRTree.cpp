#include <gtest/gtest.h>

#include "Macros.hpp"
#include "Rand.hpp"
#include "WeightedRTree.hpp"

TEST(WeightedRTree, uniformWeightsGrid) {
    const int dimension = 2;
    const int gridSize = 10;
    WeightedRTree rtree(dimension);

    VecList positions(dimension);
    std::vector<double> weights;
    std::vector<double> weightBuckets;  // no weight classes -> only one R-Tree

    positions.setSize(gridSize * gridSize);
    for (int i = 0; i < gridSize; i++) {
        for (int j = 0; j < gridSize; j++) {
            positions[i * gridSize + j][0] = i;
            positions[i * gridSize + j][1] = j;
            weights.push_back(1);
        }
    }

    rtree.updateRTree(positions, weights, weightBuckets);

    const double epsilon = 10e-6;
    CVecRef p = positions[3 * gridSize + 3];
    double weight = 1;

    // https://en.wikipedia.org/wiki/Gauss_circle_problem
    double radius = 1 + epsilon;
    std::vector<NodeId> result;
    VecBuffer<2> rTreeBuffer(dimension);
    rtree.getNodesWithinWeightedDistance(p, weight, radius, result, rTreeBuffer);
    EXPECT_EQ(result.size(), 5);

    radius = std::sqrt(2) + epsilon;
    result.clear();
    rtree.getNodesWithinWeightedDistance(p, weight, radius, result, rTreeBuffer);
    EXPECT_EQ(result.size(), 9);

    radius = 2 + epsilon;
    result.clear();
    rtree.getNodesWithinWeightedDistance(p, weight, radius, result, rTreeBuffer);
    EXPECT_EQ(result.size(), 13);

    radius = 3 + epsilon;
    result.clear();
    rtree.getNodesWithinWeightedDistance(p, weight, radius, result, rTreeBuffer);
    EXPECT_EQ(result.size(), 29);
}

TEST(WeightedRTree, uniformWeightsRandom) {
    const int dimension = 16;
    const int numPoints = 1000;
    const int numQueries = 100;
    VecBuffer<1> buffer(dimension);
    TmpVec<0> tmpVec(buffer, 0.0);

    WeightedRTree rtree(dimension);

    VecList positions(dimension);
    std::vector<double> weights;
    std::vector<double> weightBuckets;  // no weight classes -> only one R-Tree

    positions.setSize(numPoints);
    for (int i = 0; i < numPoints; i++) {
        for (int j = 0; j < dimension; j++) {
            positions[i][j] = Rand::randomDouble(0.0, 1.0);
        }
        weights.push_back(1);
    }

    rtree.updateRTree(positions, weights, weightBuckets);

    const double radius = 0.3;
    VecBuffer<2> rTreeBuffer(dimension);

    for (int i = 0; i < numQueries; i++) {
        CVecRef p = positions[Rand::randomInt(0, numPoints - 1)];
        double weight = 1;
        std::vector<NodeId> result;
        rtree.getNodesWithinWeightedDistance(p, weight, radius, result, rTreeBuffer);

        for (int i = 0; i < numPoints; i++) {
            tmpVec = p - positions[i];
            double dist = tmpVec.norm();
            if (std::find(result.begin(), result.end(), i) != result.end()) {
                EXPECT_LE(dist, radius);
            } else {
                EXPECT_GT(dist, radius);
            }
        }
    }
}

TEST(WeightedRTree, Random) {
    const int dimension = 16;
    const int numPoints = 1000;
    const int numQueries = 200;
    const double doubleFactor = 2.0;
    VecBuffer<1> buffer(dimension);
    TmpVec<0> tmpVec(buffer, 0.0);

    WeightedRTree rtree(dimension);

    VecList positions(dimension);
    std::vector<double> weights;

    positions.setSize(numPoints);
    for (int i = 0; i < numPoints; i++) {
        for (int j = 0; j < dimension; j++) {
            positions[i][j] = Rand::randomDouble(0.0, 1.0);
        }
        weights.push_back(std::pow(10, Rand::randomDouble(1e-5, 10.0)));
    }

    std::vector<double> weightBuckets = WeightedRTree::getDoublingWeightBuckets(weights, doubleFactor);
    const double minWeight = *std::min_element(weights.begin(), weights.end());
    const double maxWeight = *std::max_element(weights.begin(), weights.end());
    for (double w : weightBuckets) {
        ASSERT_GE(w, 0);
        EXPECT_GE(w, minWeight);
        EXPECT_LE(w, maxWeight);
    }

    rtree.updateRTree(positions, weights, weightBuckets);

    const double radius = 0.3;
    VecBuffer<2> rTreeBuffer(dimension);

    for (int i = 0; i < numQueries; i++) {
        int nodeId = Rand::randomInt(0, numPoints - 1);
        CVecRef p = positions[nodeId];
        double weight = weights[nodeId];
        std::vector<NodeId> result;
        rtree.getNodesWithinWeightedDistance(p, weight, radius, result, rTreeBuffer);

        for (int i = 0; i < numPoints; i++) {
            tmpVec = p - positions[i];
            double dist = tmpVec.norm();
            int bucketID =
                std::upper_bound(weightBuckets.begin(), weightBuckets.end(), weights[i]) - weightBuckets.begin();
            double weightBucketVal;
            if (bucketID == weightBuckets.size()) {
                weightBucketVal = maxWeight;
            } else {
                weightBucketVal = weightBuckets[bucketID];
            }

            EXPECT_LE(weights[i], weightBucketVal)
                << "weight: " << weights[i] << " weightBucket: " << weightBucketVal << " bucketID: " << bucketID
                << " num buckets: " << weightBuckets.size();

            
            if (std::find(result.begin(), result.end(), i) != result.end()) {
                EXPECT_LE(dist, radius * std::pow(weight * weightBucketVal, 1.0 / (double)dimension))
                    << "dist: " << dist << " radius: " << radius << " weight: " << weight
                    << " weightBucket: " << weightBucketVal << " dimension: " << dimension;
            } else {
                EXPECT_GT(dist, radius * std::pow(weight * weightBucketVal, 1.0 / (double)dimension))
                    << "dist: " << dist << " radius: " << radius << " weight: " << weight
                    << " weightBucket: " << weightBucketVal << " dimension: " << dimension;
            }
        }
    }
}