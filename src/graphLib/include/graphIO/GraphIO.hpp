#pragma once

#include <map>

#include "Graph.hpp"

enum class GraphFileType { edgeList = 0, adjList = 1, dot = 2 };

class GraphIO {
   public:
    /**
     * Reads a graph from an edge list.
     * Every line is a pair of numbers that indicate an edge separated by the delimiter symbol.
     * Lines starting with the comment symbol are ignored.
     * Will treat the graph as undirected.
     *
     * Graph ids have to be consecutive starting from 0.
     */
    static Graph readEdgeList(std::string filePath, std::string comment = "%", std::string delimiter = " ");

    /**
     * Writes the graph to file in the edge list format.
     */
    static void writeToEdgeList(std::string filePath, const Graph &g);
};
