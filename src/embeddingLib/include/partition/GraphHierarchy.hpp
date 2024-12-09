#pragma once

#include "Graph.hpp"
#include "LabelPropagation.hpp"
#include "VecList.hpp"

using ParentPointerTree = std::vector<std::vector<NodeId>>;

struct NodeInformation {
    // pointer to next and previous layer
    int parentNode = -1;
    std::vector<int> children;  // This is kind of needed for calculating approx rep forces

    // information about all contained nodes
    int totalContainedNodes = 0;
    double nodeWeightSum = 0;  // sum of the node weights
};

struct EdgeInformation {
    // pointer to next and previous layer
    int parentEdge = 1;
    std::vector<int> children;
    int totalContainedEdges = 0;  // counts the number of edges between clusters
};

/**
 * Stores information about the graph hierarchy
 * Caches values that help to speed up the calculation of the forces
 */
class GraphHierarchy {
   public:
    GraphHierarchy(const Graph& originalGraph, LabelPropagation& coarsener);
    ~GraphHierarchy() {};

    int getNumLayers() const;
    int getLayerSize(int layer) const;

    int NUMLAYERS;
    std::vector<Graph> graphs;
    std::vector<std::vector<NodeInformation>> nodeLayers;
    std::vector<std::vector<EdgeInformation>> edgeLayers;

   private:
};
