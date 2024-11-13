#include "EmbeddedGraph.hpp"

EmbeddedGraph::EmbeddedGraph(int dimension, const Graph& simpleGraph)
    : coordinates(dimension), graph(simpleGraph), nodeWeights(graph.getNumVertices(), -1.0) {
    const int N = graph.getNumVertices();
    coordinates.setSize(N, 0.0);
}

CVecRef EmbeddedGraph::getPosition(NodeId node) const { return coordinates[node]; }

void EmbeddedGraph::setPosition(NodeId node, CVecRef coords) { coordinates[node] = coords; }

double EmbeddedGraph::getNodeWeight(NodeId node) const { return nodeWeights[node]; }

void EmbeddedGraph::setNodeWeight(NodeId node, double weight) { nodeWeights[node] = weight; }

int EmbeddedGraph::getDimension() const { return coordinates.dimension(); }

NodeId EmbeddedGraph::getNumVertices() const { return graph.getNumVertices(); }

EdgeId EmbeddedGraph::getNumEdges() const { return graph.getNumEdges(); }

std::vector<NodeId> EmbeddedGraph::getNeighbors(NodeId v) const { return graph.getNeighbors(v); }

int EmbeddedGraph::getNumNeighbors(NodeId v) const { return graph.getNumNeighbors(v); }

std::vector<EdgeId> EmbeddedGraph::getEdges(NodeId v) const { return graph.getEdges(v); }

std::vector<EdgeContent> EmbeddedGraph::getEdgeContents(NodeId v) const { return graph.getEdgeContents(v); }

NodeId EmbeddedGraph::getEdgeTarget(EdgeId e) const { return graph.getEdgeTarget(e); }

bool EmbeddedGraph::areNeighbors(NodeId v, NodeId u) const { return graph.areNeighbors(v, u); }

std::string EmbeddedGraph::toString() const {
    std::string result = "";

    result += "Graph AdjList:\n";
    for (NodeId v = 0; v < getNumVertices(); v++) {
        result += std::to_string(v) + " (" + std::to_string(getNodeWeight(v)) + "): ";

        for (EdgeId e : getEdges(v)) {
            result += std::to_string(getEdgeTarget(e)) + " ";
        }
        result += "\n";
    }

    return result;
}

const Graph& EmbeddedGraph::getGraph() const { return graph; }

std::vector<double> EmbeddedGraph::getAllNodeWeights() const { return nodeWeights; }
