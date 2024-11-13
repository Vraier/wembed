#pragma once

#include "Embedding.hpp"
#include "EvalOptions.hpp"
#include "Graph.hpp"
#include "WeightedGeometric.hpp"

struct histEntry {
    double similarity;
    double distance;

    NodeId v;
    NodeId w;
    double weightV;
    double weightW;
    bool isEdge;
};

using Histogram = std::vector<histEntry>;

struct histInfo {
    Histogram hist;
    int numEdges;
    int numNonEdges;
};

bool histComparator(const histEntry& a, const histEntry& b);

class EdgeSampler {
   public:
    static histInfo sampleHistEntries(const OptionValues& opts, const Graph& graph, Embedding& embedding);

   private:
    static histEntry getHistEntry(Embedding& embedding, NodeId v, NodeId w, bool isEdge);
    static histEntry getHistEntry(WeightedGeometric& embedding, NodeId v, NodeId w, bool isEdge);
};