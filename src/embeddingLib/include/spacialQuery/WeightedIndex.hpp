#pragma once

#include <memory>

#include "EmbedderOptions.hpp"
#include "Graph.hpp"
#include "SpacialIndex.hpp"
#include "VecList.hpp"

/**
 * Wraps a spacial index into multiple weight buckets to allow for weighted queries.
 *
 * Pairs are enumerated asymmetrically: each pair is reported exactly once, to its
 * heavier endpoint (ties towards the smaller id). A node therefore only queries its
 * own and lighter classes, with radius (w_v * classMax)^(1/d).
 */
class WeightedIndex {
   public:
    WeightedIndex(IndexType type, int dimension, double doublingFactor)
        : indexType(type), DIMENSION(dimension), doublingFactor(doublingFactor) {}

    // rebuilds the per-class indices; positions/weights are borrowed until the next update
    void update(const VecList& positions, const std::vector<double>& weights);

    // exactly the pairs v owns with weighted distance < 1
    void getOwnedRepellingPairs(NodeId v, std::vector<NodeId>& out) const;

    // heavier endpoint owns a pair, ties towards the smaller id; also used by the
    // attraction pass to cancel owned neighbor pairs out of the loss
    static bool ownsPair(double weightV, double weightU, NodeId v, NodeId u) {
        return weightV > weightU || (weightV == weightU && v < u);
    }

   private:
    void queryClass(size_t weightClass, CVecRef p, double weight, std::vector<NodeId>& output) const;

    IndexType indexType;
    int DIMENSION;
    double doublingFactor;

    // borrowed from update(); valid until the next update()
    const VecList* positions = nullptr;
    const std::vector<double>* weights = nullptr;

    std::vector<double> invExpWeights;  // w^(-1/d), for the exact threshold check

    // one index per weight class; nodes in class i have weight <= maxWeightOfClass[i],
    // so querying with the class max misses no pair
    std::vector<std::shared_ptr<SpatialIndex>> spacialIndices;
    std::vector<double> maxWeightOfClass;
    std::vector<double> classBounds;                 // upper bounds used for class assignment
    std::vector<std::vector<NodeId>> classToGlobal;  // per class: local query id -> node id
};
