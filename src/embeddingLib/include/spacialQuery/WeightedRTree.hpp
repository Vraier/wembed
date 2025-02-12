#pragma once

#include <memory>

#include "RTree.hpp"
#include "SpacialIndex.hpp"
#include "VecList.hpp"

class WeightedRTree {
   public:
    using CandidateList = std::vector<std::pair<NodeId, NodeId>>;

    WeightedRTree(int dimension) : DIMENSION(dimension) {}

    /**
     * Rebuilds all r trees by inserting the positions into the right r-tree according to the weight class.
     */
    void updateRTree(const VecList& positions, const std::vector<double>& weights,
                     const std::vector<double>& weightBuckets);

    static std::vector<double> getDoublingWeightBuckets(const std::vector<double>& weights,
                                                        double doublingFactor = 2.0);

    /**
     * Searches the trees of all classes and performs distance queries on them.
     * The distance depends on the weightclass of the tree, the weight of the node and the given radius.
     *
     * Finds all p,q, with |p-q| <= radius * (weightClass(q) * weight)^(1/d)
     */
    void getNodesWithinWeightedDistance(CVecRef p, double weight, double radius, std::vector<NodeId>& output,
                                        VecBuffer<2>& buffer) const;

    void getNodesWithinWeightedDistanceForClass(CVecRef p, double weight, double radius, size_t weight_class,
                                                std::vector<NodeId>& output, VecBuffer<2>& buffer) const;

    /**
     * Same as other method but uses infNorm/box as distance metric.
     */
    void getNodesWithinWeightedInfNormDistance(CVecRef p, double weight, double radius, std::vector<NodeId>& output,
                                               VecBuffer<2>& buffer) const;

    void getNodesWithinWeightedDistanceInfNormForClass(CVecRef p, double weight, double radius, size_t weight_class,
                                                       std::vector<NodeId>& output, VecBuffer<2>& buffer) const;

    int getNumWeightClasses() const;

   private:
    void getKNNNeighbors(int indexId, CVecRef p, int k, std::vector<NodeId>& output) const;
    void getWithinRadius(int indexId, CVecRef p, double radius, std::vector<NodeId>& output,
                         VecBuffer<2>& buffer) const;
    void getWithinBox(int indexId, CVecRef p, double radius, std::vector<NodeId>& output, VecBuffer<2>& buffer) const;

    int DIMENSION;

    // assume nodes to always have the highest possible weight in a weight class
    // this way, no node will be missed when searching for non neighbors
    std::vector<std::unique_ptr<SpatialIndex>> spacialIndices;  // one R-Tree for each weight class
    std::vector<double> maxWeightOfClass;  // nodes in tree i will have weight at most weightClasses[i]
};