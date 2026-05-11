#pragma once

#include <cstddef>

#include "Graph.hpp"
#include "VecList.hpp"
#include "WeightedIndex.hpp"

struct EmbedderParameters {
    size_t currentIteration = 0;
    bool insignificantPosChange = false;

    VecList force;
    std::vector<double> weightParameterForce;
    std::vector<NodeId> indexToGraphMap;
    WeightedIndex currentWeightedIndex;

    explicit EmbedderParameters(const uint32_t graphSize, const int32_t dimension, const IndexType indexType)
                              : force(dimension, graphSize),
                                weightParameterForce(graphSize),
                                indexToGraphMap(graphSize),
                                currentWeightedIndex(indexType, dimension)
    {

    }

    void nextStep() {
        currentIteration++;

        force.setAll(0);
        std::ranges::fill(weightParameterForce, 0.0);
    }
};
