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
    static Graph readEdgeList(std::string filePath, std::string comment = "#", std::string delimiter = " ");

    /**
     * Writes the graph to file in the edge list format.
     */
    static void writeToEdgeList(std::string filePath, const Graph &g);

    /**
     * Same as read Edge list, but the first line containes the numbers a,b. 
     * a+b=n and the first a nodes correspond to the first partition and the last b nodes to the second partition.
     * 
     * Sets the color in the graph according to the partitions.
     */
    static Graph readBipartiteEdgeList(std::string filePath, std::string comment = "#", std::string delimiter = " ");
};
