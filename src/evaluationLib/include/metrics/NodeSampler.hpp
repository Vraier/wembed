#pragma once

#include <memory>

#include "Embedding.hpp"
#include "Graph.hpp"
#include "WeightedGeometric.hpp"

struct nodeEntry {
    NodeId v;
    int degV;

    flt_t deg_precision;
    flt_t average_precision;
};

/**
 * Vector of sorted pairs of edge lengths and node ids.
 * Can be used to find out how many neighbors have distance smaller than l
 */
using EdgeLengthToNode = std::vector<std::pair<flt_t, NodeId>>;

/**
 * Samples random nodes from the graph. Mainly used by the reonstruction metric.
 */
class NodeSampler {
   public:
    static std::vector<nodeEntry> sampleHistEntries(const Graph &graph, std::shared_ptr<Embedding> embedding, int numNodeSamples);

   private:
    static std::vector<flt_t> getPrecisionsForNode(NodeId v, const EdgeLengthToNode &distances, const std::vector<bool> &isNeighbor);
    static std::vector<flt_t> getRecallsForNode(NodeId v, int deg, const EdgeLengthToNode &distances, const std::vector<bool> &isNeighbor);
    static flt_t getAveragePrecision(NodeId v, const EdgeLengthToNode &distances, const std::vector<flt_t> &precisions, const std::vector<flt_t> &recalls, const std::vector<bool> &isNeighbor);
};