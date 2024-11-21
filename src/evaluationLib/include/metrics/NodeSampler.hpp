#pragma once

#include <memory>

#include "Embedding.hpp"
#include "Graph.hpp"
#include "WeightedGeometric.hpp"

struct nodeEntry {
    NodeId v;
    int degV;

    double deg_precision;
    double average_precision;
};

/**
 * Vector of sorted pairs of edge lengths and node ids.
 * Can be used to find out how many neighbors have distance smaller than l
 */
using EdgeLengthToNode = std::vector<std::pair<double, NodeId>>;

/**
 * Samples random nodes from the graph. Mainly used by the reonstruction metric.
 */
class NodeSampler {
   public:
    static std::vector<nodeEntry> sampleHistEntries(const Graph &graph, std::shared_ptr<Embedding> embedding, double nodeSampleFraction);
    static double averageFromVector(const std::vector<double> &values);

   private:
    static std::vector<double> getPrecisionsForNode(NodeId v, const EdgeLengthToNode &distances, const std::vector<bool> &isNeighbor);
    static std::vector<double> getRecallsForNode(NodeId v, int deg, const EdgeLengthToNode &distances, const std::vector<bool> &isNeighbor);
    static double getAveragePrecision(NodeId v, const EdgeLengthToNode &distances, const std::vector<double> &precisions, const std::vector<double> &recalls, const std::vector<bool> &isNeighbor);
};