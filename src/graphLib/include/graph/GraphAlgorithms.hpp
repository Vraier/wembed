#pragma once

#include "Graph.hpp"

class GraphAlgo {
   public:

    static int getNumberOfConnectedComponents(const Graph &g);
    static bool isConnected(const Graph &g);

    static Graph getLargestComponent(const Graph &g);

    /**
     * Mapping from the new node id to the old node ids
     */
    static std::pair<Graph, std::vector<NodeId>> getLargestComponentWithMapping(const Graph &g);

    /**
     * returns the connected components of a graph
     * the first array gives the id of the connected component of a node.
     * the second array gives the size of the connected component.
     */
    static std::pair<std::vector<int>, std::vector<int>> calculateComponentId(const Graph &g);
    /**
     * returns a new graph the only contains the larges connected component of the given graph.
     * The indices of the new graph are mapped to fit between 0..newSize
     */
    //static Graph getLargestComponent(Graph &unconnected);

    /**
     * returns a new graph that contracts the nodes in g according to the mapping in clusterId
     */
    static std::pair<Graph, std::vector<EdgeId>> coarsenGraph(const Graph &g, const std::vector<NodeId> &clusterId);


    /**
     * Calculates the length of a shortest path from origin to all other nodes
     */
    static std::vector<int> calculateShortestPaths(const Graph &g, NodeId origin);

    /**
     * Calculates the distance of the shortest path between all pairs of nodes
     */
    static std::vector<std::vector<int>> calculateAllPairShortestPaths(const Graph &g);
};