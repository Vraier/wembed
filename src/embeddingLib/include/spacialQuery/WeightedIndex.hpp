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
 *
 * With dynamicBuffer > 0 rebuilds are amortized over iterations (rembed's DynamicQuery):
 * after a rebuild, queries run once with radii inflated by the buffer and cache each
 * node's candidate list; while the accumulated movement (2 * maxDisplacement per step)
 * stays below the buffer, iterations only re-filter the cached lists against current
 * positions. 
 */
class WeightedIndex {
   public:
    WeightedIndex(IndexType type, int dimension, double doublingFactor, double dynamicBuffer,
                  double minExpectedReuses = 2.0)
        : indexType(checkedIndexType(type, dimension)),
          DIMENSION(dimension),
          doublingFactor(doublingFactor),
          dynamicBuffer(dynamicBuffer),
          minExpectedReuses(minExpectedReuses) {}

    // The sprk tree needs Rust at build time and supports 2 to 16 dimensions. Aborts if it
    // is requested where it cannot be used: the slower KD-tree has to be chosen explicitly.
    static IndexType checkedIndexType(IndexType type, int dimension);

    // refreshes the index; maxDisplacement is the largest single-node movement since the
    // previous call (pass infinity after any discontinuous position/weight change)
    void update(const VecList& positions, const std::vector<double>& weights, double maxDisplacement);

    // exactly the pairs v owns with weighted distance < 1
    void getOwnedRepellingPairs(NodeId v, std::vector<NodeId>& out);

    // heavier endpoint owns a pair, ties towards the smaller id; also used by the
    // attraction pass to cancel owned neighbor pairs out of the loss
    static bool ownsPair(double weightV, double weightU, NodeId v, NodeId u) {
        return weightV > weightU || (weightV == weightU && v < u);
    }

    size_t numUpdates() const { return updateCalls; }
    size_t numRebuilds() const { return rebuildCalls; }

   private:
    enum class QueryMode {
        Plain,  // tight radii, no caching, default when dynamicBuffer == 0
        Fill,   // radii inflated by dynamicBuffer, cache the candidates
        Reuse   // answer from the cached lists; the trees are not touched
    };

    void rebuildClasses();
    void queryClass(size_t weightClass, CVecRef p, double weight, double radiusSlack,
                    std::vector<NodeId>& output) const;

    IndexType indexType;
    int DIMENSION;
    double doublingFactor;
    double dynamicBuffer;
    // a fill (inflated radii) only pays off if the buffer survives a few steps of the
    // current movement; below that, query tight like before
    double minExpectedReuses;

    QueryMode mode = QueryMode::Plain;
    double remainingBudget = -1.0;
    size_t updateCalls = 0;
    size_t rebuildCalls = 0;
    // per node: owned candidates within the inflated radius at fill time.
    // INVARIANT: cachedPairs[v] is only touched by the thread querying v
    std::vector<std::vector<NodeId>> cachedPairs;

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
