#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

typedef int NodeId;
typedef int EdgeId;

struct NodeContent {
    EdgeId firstEdge;
};

struct EdgeContent {
    NodeId neighbour;
};

/**
 * Class representing a static undirected graph.
 * All vertex identifiers must be between 0 and n-1, where n is the number of vertices.
 * Vertex identifiers should be consecutive.
 */
class Graph {
   public:
    Graph() { setSize(0, 0); };
    Graph(std::map<int, std::set<int>> &map) {
        constructFromMap(map);
        setUniqueColors();
    };
    Graph(std::vector<std::pair<int, int>> &edges) {
        constructFromEdges(edges);
        setUniqueColors();
    };
    ~Graph() {};

    void setUniqueColors();
    void setColors(std::vector<int> &colors);

    // global information
    NodeId getNumVertices() const;
    EdgeId getNumEdges() const;

    // neighborhood information
    std::vector<EdgeId> getEdges(NodeId v) const;
    std::vector<NodeId> getNeighbors(NodeId v) const;
    int getNumNeighbors(NodeId v) const;
    std::vector<EdgeContent> getEdgeContents(NodeId v) const;
    NodeId getEdgeTarget(EdgeId e) const;
    bool areNeighbors(NodeId v, NodeId u) const;  // runtime linear in deg(v)
    bool areInSameColorClass(NodeId v, NodeId u) const;

    /**
     * returns a string representation of the graph
     */
    std::string toString() const;

    /*
     * Functions to modify the internal state of the graph.
     * Don't use these. the constructor should be enough.
     */
    void setSize(NodeId n, EdgeId m);
    void nextNode();
    void addEdge(NodeId target);

   private:
    /*
     * constructs a graph from a map
     * the map contains a set of neighbors for each vertex.
     * Node ids should start by 0 and be consecutive.
     */
    void constructFromMap(const std::map<int, std::set<int>> &map);

    /**
     * Constructs a graph from a set of edges.
     * Node ids should start by 0 and be consecutive.
     */
    void constructFromEdges(const std::vector<std::pair<NodeId, NodeId>> &edges);

    std::vector<NodeContent> nodes;
    std::vector<EdgeContent> edges;
    std::vector<int> colors;
    NodeId currentNodeBuildingId = 0;
    EdgeId currentEdgeBuildingId = 0;
};
