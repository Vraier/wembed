#pragma once

#include <memory>

#include "EmbedderOptions.hpp"
#include "Graph.hpp"
#include "SpacialIndex.hpp"
#include "VecList.hpp"

class WeightedIndex {
   public:
    using CandidateList = std::vector<std::pair<NodeId, NodeId>>;

    WeightedIndex(IndexType type, int dimension) : indexType(type), DIMENSION(dimension) {}

    /**
     * Rebuilds all r indices by inserting the positions into the right index according to the weight class.
     */
    void updateIndices(const VecList& positions, const std::vector<double>& weights,
                       const std::vector<double>& weightBuckets);

    /**
     * Returns the weight buckets used for index construction. The smalles weight has the sice doublingFactor*minWeight.
     * Afterwards the weights are increased by the factor doublingFactor until the maximum weight is surpassed.
     */
    static std::vector<double> getDoublingWeightBuckets(const std::vector<double>& weights,
                                                        double doublingFactor = 2.0);

    /**
     * Searches the indices of all classes and performs distance queries on them.
     * The distance depends on the weightclass of the index, the weight of the node and the given radius.
     *
     * Finds all p,q, with |p-q| <= radius * (weightClass(q) * weight)^(1/d)
     */
    void getNodesWithinWeightedDistance(CVecRef p, double weight, double radius, std::vector<NodeId>& output,
                                        VecBuffer<2>& buffer) const;

    /**
     * Same as other method but uses infNorm/box as distance metric.
     */
    void getNodesWithinWeightedInfNormDistance(CVecRef p, double weight, double radius, std::vector<NodeId>& output,
                                               VecBuffer<2>& buffer) const;

    int getNumWeightClasses() const;
    int getIndexDimension() const;
    std::vector<double> getWeightClasses() const;

   private:
    void getKNNNeighbors(int indexId, CVecRef p, int k, std::vector<NodeId>& output) const;
    void getWithinRadius(int indexId, CVecRef p, double radius, std::vector<NodeId>& output,
                         VecBuffer<2>& buffer) const;
    void getWithinBox(int indexId, CVecRef p, double radius, std::vector<NodeId>& output, VecBuffer<2>& buffer) const;

    // Helper methods for the corresponding getNodesWithinWeightedDistance methods
    void getNodesWithinWeightedDistanceForClass(CVecRef p, double weight, double radius, size_t weight_class,
                                                std::vector<NodeId>& output, VecBuffer<2>& buffer) const;
    void getNodesWithinWeightedDistanceInfNormForClass(CVecRef p, double weight, double radius, size_t weight_class,
                                                       std::vector<NodeId>& output, VecBuffer<2>& buffer) const;

    IndexType indexType;
    int DIMENSION;

    // assume nodes to always have the highest possible weight in a weight class
    // this way, no node will be missed when searching for non neighbors
    std::vector<std::shared_ptr<SpatialIndex>> spacialIndices;  // one index for each weight class
    std::vector<double> maxWeightOfClass;  // nodes in index will have weight at most weightClasses[i]
};