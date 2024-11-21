#pragma once

#include <memory>

#include "Embedding.hpp"
#include "Graph.hpp"
#include "WeightedGeometric.hpp"

struct histEntry {
    double similarity;

    NodeId v;
    NodeId w;
    bool isEdge;
};

struct histInfo {
    std::vector<histEntry> histogramm;
    int numEdges;
    int numNonEdges;
};

bool histComparator(const histEntry& a, const histEntry& b);

/**
 * Used to sample random edges and non edges from the graph.
 * Will construct a list of all sampled pairs containing usefull information for thurther processing.
 * 
 * Mainly used by the F1-Score metric.
 */
class EdgeSampler {
   public:
    static histInfo sampleHistEntries(const Graph& graph, std::shared_ptr<Embedding> embedding, double sampleingScale);
};