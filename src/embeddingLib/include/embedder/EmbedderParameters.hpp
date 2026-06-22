#pragma once

#include <cstddef>

#include "Graph.hpp"
#include "VecList.hpp"
#include "WeightedIndex.hpp"

/**
 * struct for holding parameters needed to compute an embedding
 */
struct EmbedderParameters {
    size_t currentIteration = 0;
    bool insignificantPosChange = false;

    VecList force;
    std::vector<NodeId> indexToGraphMap;
    WeightedIndex currentWeightedIndex;

    explicit EmbedderParameters(const uint32_t graphSize, const int32_t dimension, const IndexType indexType)
                              : force(dimension, graphSize),
                                indexToGraphMap(graphSize),
                                currentWeightedIndex(indexType, dimension)
    {

    }

    void nextStep() {
        currentIteration++;

        force.setAll(0);
    }

    //TODO: pass centreOfGrvity
    void nextStep(CVecRef centreOfGrav) {
        currentIteration++;

        force.setAll(centreOfGrav);
    }
};
