#include "Graph.hpp"

#include <fstream>
#include <iostream>
#include <numeric>

#include "Macros.hpp"

void Graph::setSize(NodeId n, EdgeId m) {
    // LOG_DEBUG( "Setting the size of a graph to n=" << n << " m=" << m);
    nodes.resize(n + 1);
    edges.resize(m * 2);
}

void Graph::nextNode() {
    ASSERT(currentNodeBuildingId < nodes.size(),
           std::to_string(nodes.size()) + " " + std::to_string(currentNodeBuildingId));
    currentNodeBuildingId++;
    nodes[currentNodeBuildingId].firstEdge = currentEdgeBuildingId;
}

void Graph::addEdge(NodeId target) {
    ASSERT(currentEdgeBuildingId < edges.size(),
           std::to_string(edges.size()) + " " + std::to_string(currentEdgeBuildingId));
    ASSERT(target < nodes.size(), std::to_string(nodes.size()) + " " + std::to_string(target));
    edges[currentEdgeBuildingId].neighbour = target;
    currentEdgeBuildingId++;
}

std::vector<NodeId> Graph::getNeighbors(NodeId v) const {
    int numNeighbors = nodes[v + 1].firstEdge - nodes[v].firstEdge;
    std::vector<NodeId> result(numNeighbors);
    int i = 0;
    for (EdgeId e = nodes[v].firstEdge; e < nodes[v + 1].firstEdge; e++) {
        result[i++] = (edges[e].neighbour);
    }
    return result;
}

int Graph::getNumNeighbors(NodeId v) const {
    int numNeighbors = nodes[v + 1].firstEdge - nodes[v].firstEdge;
    return numNeighbors;
}

std::vector<EdgeId> Graph::getEdges(NodeId v) const {
    int numNeighbors = nodes[v + 1].firstEdge - nodes[v].firstEdge;
    std::vector<EdgeId> result(numNeighbors);
    int i = 0;
    for (EdgeId e = nodes[v].firstEdge; e < nodes[v + 1].firstEdge; e++) {
        result[i++] = e;
    }
    return result;
}

std::vector<EdgeContent> Graph::getEdgeContents(NodeId v) const {
    int numNeighbors = nodes[v + 1].firstEdge - nodes[v].firstEdge;
    std::vector<EdgeContent> result(numNeighbors);
    int i = 0;
    for (EdgeId e = nodes[v].firstEdge; e < nodes[v + 1].firstEdge; e++) {
        result[i++] = edges[e];
    }
    return result;
}

NodeId Graph::getEdgeTarget(EdgeId e) const { return edges[e].neighbour; }

bool Graph::areNeighbors(NodeId v, NodeId u) const {
    int numNeighborsV = nodes[v + 1].firstEdge - nodes[v].firstEdge;
    int numNeighborsU = nodes[u + 1].firstEdge - nodes[u].firstEdge;

    // find the node with less neighbors
    if (numNeighborsV > numNeighborsU) {
        std::swap(v, u);
    }

    // iterate over the neighbors of the smaller node and check if the other node is among them
    for (EdgeId e = nodes[v].firstEdge; e < nodes[v + 1].firstEdge; e++) {
        if (edges[e].neighbour == u) {
            return true;
        };
    }
    return false;
}

bool Graph::areInSameColorClass(NodeId v, NodeId u) const { return colors[v] == colors[u]; }

void Graph::constructFromMap(const std::map<int, std::set<int>>& map) {
    LOG_DEBUG("Constructing graph from map");
    // make map symmetric
    std::map<NodeId, std::set<NodeId>> symmetric_map = map;
    for (auto iter = symmetric_map.begin(); iter != symmetric_map.end(); ++iter) {
        for (NodeId u : iter->second) {
            symmetric_map[u].insert(iter->first);
            symmetric_map[iter->first].insert(u);
        }
    }

    // number of nodes = largest id of a node +1 (for 0 node)
    int n = symmetric_map.empty() ? 0 : symmetric_map.rbegin()->first + 1;
    ASSERT(n >= 0);
    if (n != symmetric_map.size()) {
        LOG_WARNING(
            "The map does not contain consecutive node ids. This may lead to unexpected behavior. Filling up missing "
            "nodes.");
    }

    int m = 0;
    for (auto iter = symmetric_map.begin(); iter != symmetric_map.end(); ++iter) {
        m += iter->second.size();
    }
    setSize(n, m / 2);
    LOG_DEBUG("Constructing graph from map with n=" << n << " m=" << m / 2);

    int currentNode = 0;
    bool firstWarning = true;
    for (auto iter = symmetric_map.begin(); iter != symmetric_map.end(); ++iter) {
        while (iter->first != currentNode) {
            //  the node does not exist in the map
            //  add nodes until we reach a node that has edges
            nextNode();
            currentNode++;
        }
        for (NodeId u : iter->second) {
            if (u == iter->first && firstWarning) {
                LOG_WARNING("Node " + std::to_string(u) + " is connected to itself. Self loops are ignored.");
                firstWarning = false;
                continue;
            }
            addEdge(u);
        }
        nextNode();
        currentNode++;
    }

    ASSERT(currentNodeBuildingId == nodes.size() - 1,
           std::to_string(currentNodeBuildingId) + " " + std::to_string(nodes.size()));
    ASSERT(currentEdgeBuildingId == edges.size(),
           std::to_string(currentEdgeBuildingId) + " " + std::to_string(edges.size()));
}

void Graph::constructFromEdges(const std::vector<std::pair<NodeId, NodeId>>& edges) {
    LOG_DEBUG("Converting vector to map");
    std::map<NodeId, std::set<NodeId>> set;
    for (auto e : edges) {
        set[e.first].insert(e.second);
        set[e.second].insert(e.first);
    }

    constructFromMap(set);
}

void Graph::setUniqueColors() {
    std::vector<int> colors(getNumVertices());  // colors from 0 to n-1
    std::iota(colors.begin(), colors.end(), 0);
    setColors(colors);
}

void Graph::setColors(std::vector<int>& colors) {
    ASSERT(colors.size() == getNumVertices());
    this->colors = colors;
}

NodeId Graph::getNumVertices() const { return nodes.size() - 1; }

EdgeId Graph::getNumEdges() const { return edges.size() / 2; }

std::string Graph::toString() const {
    std::string result = "";

    result += "Graph AdjList:\n";
    for (NodeId v = 0; v < getNumVertices(); v++) {
        result += std::to_string(v) + ": ";

        for (auto e : getEdgeContents(v)) {
            result += std::to_string(e.neighbour) + " ";
        }
        result += "\n";
    }

    return result;
}