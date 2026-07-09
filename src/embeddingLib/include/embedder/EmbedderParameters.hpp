#pragma once

#include <cstddef>

#include "Graph.hpp"
#include "SprkQueries.hpp"
#include "VecList.hpp"
#include "WeightedIndex.hpp"

struct EmbedderParameters {
    size_t currentIteration = 0;
    bool insignificantPosChange = false;

    VecList force;
    std::vector<NodeId> indexToGraphMap;

    explicit EmbedderParameters(const uint32_t graphSize, const int32_t dimension, const IndexType indexType)
                              : force(dimension, graphSize),
                                indexToGraphMap(graphSize)
    {
        //TODO: Initialize spatial index
    }

    void nextStep() {
        currentIteration++;

        force.setAll(0);
    }
};
