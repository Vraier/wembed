#pragma once

#include "Embedding.hpp"
#include "EvalOptions.hpp"
#include "Graph.hpp"
#include "WeightedGeometric.hpp"

struct nodeEntry {
    NodeId v;
    int degV;
    // double weightV;

    double deg_precision;
    double average_precision;
    std::map<int, double> k_to_precision;
};

using EdgeLengthToNode = std::vector<std::pair<double, NodeId>>;
using NodeHistogram = std::vector<nodeEntry>;

class NodeSampler {
   public:
    static NodeHistogram sampleHistEntries(const OptionValues &opts, const Graph &graph, Embedding &embedding, const std::vector<int> &kVals);
    static double averageFromVector(const std::vector<double> &values);

   private:
    static std::vector<double> getPrecisionsForNode(NodeId v, const EdgeLengthToNode &distances, const std::vector<bool> &isNeighbor);
    static std::vector<double> getRecallsForNode(NodeId v, int deg, const EdgeLengthToNode &distances, const std::vector<bool> &isNeighbor);
    static double getAveragePrecision(NodeId v, const EdgeLengthToNode &distances, const std::vector<double> &precisions, const std::vector<double> &recalls, const std::vector<bool> &isNeighbor);
};