#pragma once

#include "Graph.hpp"
#include "VecList.hpp"

class EmbeddedGraph {
   public:
    EmbeddedGraph(int dimension, const Graph& simpleGraph);

    int getDimension() const;

    CVecRef getPosition(NodeId node) const;
    void setPosition(NodeId node, CVecRef coords);

    double getNodeWeight(NodeId node) const;
    void setNodeWeight(NodeId node, double weight);

    // returns the weight of every node
    std::vector<double> getAllNodeWeights() const;

    // Functions wrapping graph
    NodeId getNumVertices() const;
    EdgeId getNumEdges() const;
    std::vector<NodeId> getNeighbors(NodeId v) const;
    int getNumNeighbors(NodeId v) const;
    std::vector<EdgeId> getEdges(NodeId v) const;
    std::vector<EdgeContent> getEdgeContents(NodeId v) const;
    NodeId getEdgeTarget(EdgeId e) const;
    bool areNeighbors(NodeId v, NodeId u) const;

    std::string toString() const;

    const Graph& getGraph() const;

    VecList coordinates;

   private:
    Graph graph;

    std::vector<double> nodeWeights;  // needed for heterogeneous graphs
};
