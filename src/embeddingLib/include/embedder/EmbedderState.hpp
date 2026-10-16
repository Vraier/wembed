#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "Graph.hpp"
#include "VecList.hpp"
#include "WeightedIndex.hpp"

/**
 * All mutable state of a single embedding run: the current layout, the per-step
 * working buffers, and the observables of the most recent step (read back through
 * EmbedderInterface's accessors and by the stopping-criterion monitors).
 * Configuration lives in EmbedderOptions; this is everything that changes as the
 * embedding progresses.
 */
struct EmbedderState {
    // Current layout
    VecList<> currentPositions;
    std::vector<flt_t> currentWeights;
    std::vector<int32_t> sortedNodeIDs;  // node IDs sorted by descending weight

    // Per-step working buffers
    size_t currentIteration = 0;
    VecList<> force;
    std::vector<NodeId> indexToGraphMap;
    WeightedIndex currentWeightedIndex;

    // Observables of the most recent step
    flt_t lastAttractLoss = 0.0;
    flt_t lastRepelLoss = 0.0;
    flt_t lastLearningRate = 0.0;
    flt_t lastRelDisplacement = 0.0;     // rate the displacement stop watches
    flt_t lastRelLossImprovement = 0.0;  // rate(t) the loss stop watches
    flt_t stepSeconds = 0.0;             // time spent inside calculateStep so far
    // consumed by the dynamic-query budget in WeightedIndex; infinity forces a rebuild
    flt_t lastMaxDisplacement = std::numeric_limits<flt_t>::infinity();

    EmbedderState(uint32_t graphSize, int32_t dimension, IndexType indexType, flt_t doublingFactor,
                  flt_t dynamicQueryBuffer, flt_t dynamicQueryMinReuses)
        : currentPositions(dimension, graphSize),
          currentWeights(graphSize),
          sortedNodeIDs(graphSize),
          force(dimension, graphSize),
          currentWeightedIndex(indexType, dimension, doublingFactor, dynamicQueryBuffer, dynamicQueryMinReuses) {}

    // Reset the per-step accumulators before a new step.
    void nextStep() {
        currentIteration++;
        force.setAll(0);
        lastAttractLoss = 0.0;
        lastRepelLoss = 0.0;
    }
};
