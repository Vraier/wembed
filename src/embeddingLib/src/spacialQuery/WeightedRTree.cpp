#include "WeightedRTree.hpp"

#include <iostream>

using rTreeValue = std::pair<CVecRef, NodeId>;

void WeightedRTree::updateRTree(const VecList& positions, const std::vector<double>& weights,
                                const std::vector<double>& weightBuckets) {
    ASSERT(positions.size() == weights.size(), "Positions and weights must have the same size");
    ASSERT(std::is_sorted(weightBuckets.begin(), weightBuckets.end()), "Weight buckets must be sorted");

    double maxWeight = 0;
    if (!weightBuckets.empty()) {
        maxWeight = weightBuckets.back();
    }

    std::vector<std::vector<rTreeValue>> valuesPerBucket(weightBuckets.size() + 1);
    for (NodeId v = 0; v < positions.size(); v++) {
        double weight = weights[v];
        if (weight > maxWeight) {
            maxWeight = weight;
        }
        auto it = std::upper_bound(weightBuckets.begin(), weightBuckets.end(), weight) - weightBuckets.begin();
        valuesPerBucket[it].push_back(std::make_pair(positions[v], v));
    }

    maxWeightOfClass = weightBuckets;
    maxWeightOfClass.push_back(maxWeight);

    spacialIndices.clear();
    for (int i = 0; i < weightBuckets.size() + 1; i++) {
        std::unique_ptr<SpatialIndex> index = std::make_unique<RTree>(std::move(valuesPerBucket[i]), DIMENSION);
        spacialIndices.push_back(std::move(index));
    }
}

std::vector<double> WeightedRTree::getDoublingWeightBuckets(const std::vector<double>& weights, double doublingFactor) {
    double minWeight = *std::min_element(weights.begin(), weights.end());
    double maxWeight = *std::max_element(weights.begin(), weights.end());

    std::vector<double> buckets;
    double currentWeight = minWeight * doublingFactor;
    while (currentWeight < maxWeight) {
        buckets.push_back(currentWeight);
        currentWeight *= doublingFactor;
    }

    return buckets;
}

void WeightedRTree::getNodesWithinWeightedDistance(CVecRef p, double weight, double radius, std::vector<NodeId>& output,
                                                   VecBuffer<2>& buffer) const {
    ASSERT(output.empty());
    for (int i = 0; i < maxWeightOfClass.size(); i++) {
        getNodesWithinWeightedDistanceForClass(p, weight, radius, i, output, buffer);
    }
}

void WeightedRTree::getNodesWithinWeightedDistanceForClass(CVecRef p, double weight, double radius, size_t weight_class,
                                                           std::vector<NodeId>& output, VecBuffer<2>& buffer) const {
    ASSERT(spacialIndices.size() == maxWeightOfClass.size(), "Indices and weight classes must have the same size");
    ASSERT(weight_class < maxWeightOfClass.size());

    double maxWeight = maxWeightOfClass[weight_class];
    double queryRadius = radius * Toolkit::myPow(weight * maxWeight, 1.0 / (double)DIMENSION);
    getWithinRadius(weight_class, p, queryRadius, output, buffer);
}

void WeightedRTree::getNodesWithinWeightedInfNormDistance(CVecRef p, double weight, double radius,
                                                          std::vector<NodeId>& output, VecBuffer<2>& buffer) const {
    ASSERT(output.empty());
    for (int i = 0; i < maxWeightOfClass.size(); i++) {
        getNodesWithinWeightedDistanceInfNormForClass(p, weight, radius, i, output, buffer);
    }
}

void WeightedRTree::getNodesWithinWeightedDistanceInfNormForClass(CVecRef p, double weight, double radius,
                                                                  size_t weight_class, std::vector<NodeId>& output,
                                                                  VecBuffer<2>& buffer) const {
    ASSERT(spacialIndices.size() == maxWeightOfClass.size(), "Indices and weight classes must have the same size");
    ASSERT(weight_class < maxWeightOfClass.size());

    double maxWeight = maxWeightOfClass[weight_class];
    double queryRadius = radius * Toolkit::myPow(weight * maxWeight, 1.0 / (double)DIMENSION);
    getWithinBox(weight_class, p, queryRadius, output, buffer);
}

int WeightedRTree::getNumWeightClasses() const { return maxWeightOfClass.size(); }

void WeightedRTree::getKNNNeighbors(int indexId, CVecRef p, int k, std::vector<NodeId>& output) const {
    ASSERT(p.dimension() == DIMENSION);
    spacialIndices[indexId]->query_nearest(p, k, output);
}

void WeightedRTree::getWithinRadius(int indexId, CVecRef p, double radius, std::vector<NodeId>& output,
                                    VecBuffer<2>& buffer) const {
    ASSERT(p.dimension() == DIMENSION);
    ASSERT(radius > 0);

    // TmpVec<0> min_corner(buffer);
    // TmpVec<1> max_corner(buffer);
    // min_corner = p;
    // max_corner = p;
    // for (int i = 0; i < DIMENSION; i++) {
    //     min_corner[i] -= radius;
    //     max_corner[i] += radius;
    // }
    spacialIndices[indexId]->query_sphere(p, radius, output);
}

void WeightedRTree::getWithinBox(int indexId, CVecRef p, double radius, std::vector<NodeId>& output,
                                 VecBuffer<2>& buffer) const {
    ASSERT(p.dimension() == DIMENSION);
    ASSERT(radius > 0);

    TmpVec<0> min_corner(buffer);
    TmpVec<1> max_corner(buffer);
    min_corner = p;
    max_corner = p;
    for (int i = 0; i < DIMENSION; i++) {
        min_corner[i] -= radius;
        max_corner[i] += radius;
    }
    spacialIndices[indexId]->query_box(min_corner.erase(), max_corner.erase(), output);
}
