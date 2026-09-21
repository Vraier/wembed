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
    VecList currentPositions;
    std::vector<double> currentWeights;
    std::vector<int32_t> sortedNodeIDs;  // node IDs sorted by descending weight

    // Per-step working buffers
    size_t currentIteration = 0;
    VecList force;
    WeightedIndex currentWeightedIndex;

    // Observables of the most recent step
    double lastAttractLoss = 0.0;
    double lastRepelLoss = 0.0;
    double lastLearningRate = 0.0;
    double lastRelDisplacement = 0.0;     // rate the displacement stop watches
    double lastRelLossImprovement = 0.0;  // rate(t) the loss stop watches
    // consumed by the dynamic-query budget in WeightedIndex; infinity forces a rebuild
    double lastMaxDisplacement = std::numeric_limits<double>::infinity();

    EmbedderState(uint32_t graphSize, int32_t dimension, IndexType indexType, double doublingFactor,
                  double dynamicQueryBuffer, double dynamicQueryMinReuses)
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
