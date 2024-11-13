#include "GeometricGraphSampler.hpp"

#include <iostream>
#include <unordered_map>

#include "GraphAlgorithms.hpp"
#include "Macros.hpp"
#include "Rand.hpp"

GraphCoordinatesPair GeometricGraphSampler::generateRandomGraphWithCoordinates(int n) {
    double gridSize = std::sqrt(n);
    double radius = std::sqrt(20.0 / M_PI);
    return generateRandomGraph(n, gridSize, radius);
}

Graph GeometricGraphSampler::generateRandomGraph(int n) { return generateRandomGraphWithCoordinates(n).first; }

GraphCoordinatesPair GeometricGraphSampler::generateRandomGraph(int n, double gridSize, double radius) {
    LOG_INFO("Constructing random graph...");

    // sample random coordinates
    std::vector<std::vector<double>> coords(n);
    for (int i = 0; i < n; i++) {
        coords[i] = std::vector<double>(2);
        coords[i][0] = Rand::randomDouble(0, gridSize);
        coords[i][1] = Rand::randomDouble(0, gridSize);
    }

    // build the graph (quadratic)
    std::map<int, std::set<int>> graphMap;
    for (int v = 0; v < n; v++) {
        if (graphMap.find(v) == graphMap.end()) {
            graphMap[v] = std::set<int>();
        }
        // check all higher nodes
        for (int u = v + 1; u < n; u++) {
            double distance = std::sqrt((coords[u][0] - coords[v][0]) * (coords[u][0] - coords[v][0]) +
                                        (coords[u][1] - coords[v][1]) * (coords[u][1] - coords[v][1]));
            if (distance < radius) {
                graphMap[v].insert(u);
                graphMap[u].insert(v);
            }
        }
    }
    Graph g(graphMap);

    LOG_INFO("Finished construction");
    ASSERT(g.getNumVertices() == coords.size());
    GraphCoordinatesPair graphCoords = std::make_pair(g, coords);
    return findLargesConnectedComponent(graphCoords);
}

GraphCoordinatesPair GeometricGraphSampler::findLargesConnectedComponent(GraphCoordinatesPair& graphCoords) {
    Graph unconnected = graphCoords.first;
    std::vector<std::vector<double>> unconnectedCoords = graphCoords.second;
    auto cc = GraphAlgo::calculateComponentId(unconnected);

    std::vector<int> connectedComponent = cc.first;
    std::vector<int> componentSize = cc.second;

    // find larges component
    int largesComponent = -1;
    int largestSize = -1;
    for (int i = 0; i < componentSize.size(); i++) {
        if (componentSize[i] > largestSize) {
            largestSize = componentSize[i];
            largesComponent = i;
        }
    }

    // add nodes and edges to new graph
    std::map<int, std::set<int>> graphMap;
    std::vector<std::vector<double>> connectedCoords(largestSize);
    std::unordered_map<int, int> nodeIdMapping;
    int currIdCounter = 0;

    for (int v = 0; v < unconnected.getNumVertices(); v++) {
        if (connectedComponent[v] == largesComponent) {
            // node is in the largest component -> add it
            nodeIdMapping[v] = currIdCounter++;
            if (graphMap.find(nodeIdMapping[v]) == graphMap.end()) {
                graphMap[nodeIdMapping[v]] = std::set<int>();
            }

            // add all edges (to smaller nodes) to the new graph
            // i only handle smaller nodes so that the nodeIdMapping is already initialized
            for (int u : unconnected.getNeighbors(v)) {
                if (u > v) continue;
                graphMap[nodeIdMapping[u]].insert(nodeIdMapping[v]);
                graphMap[nodeIdMapping[v]].insert(nodeIdMapping[u]);
            }

            // write the coordinate to new vector
            std::vector<double> tmp = unconnectedCoords[v];
            connectedCoords[nodeIdMapping[v]] = tmp;
        }
    }
    Graph connected(graphMap);
    ASSERT(connected.getNumVertices() == connectedCoords.size());
    return GraphCoordinatesPair(connected, connectedCoords);
}
