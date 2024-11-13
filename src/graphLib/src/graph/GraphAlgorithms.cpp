#include "GraphAlgorithms.hpp"

#include <iostream>
#include <numeric>
#include <sstream>
#include <unordered_map>

#include "Graph.hpp"
#include "Macros.hpp"
#include "Toolkit.hpp"

int GraphAlgo::getNumberOfConnectedComponents(Graph &g) {
    auto idAndSize = calculateComponentId(g);
    return idAndSize.second.size();
}

bool GraphAlgo::isConnected(Graph &g) { return getNumberOfConnectedComponents(g) == 1; }

std::pair<std::vector<int>, std::vector<int>> GraphAlgo::calculateComponentId(Graph &g) {
    LOG_INFO("Finding connected components...");

    int n = g.getNumVertices();

    std::vector<int> connectedComponent(n, -1);
    std::vector<int> componentSize;

    // find the largest connected component
    int currComponent = 0;
    for (int v = 0; v < n; v++) {
        // the node is already in a component
        if (connectedComponent[v] != -1) {
            continue;
        } else {
            // do BFS to calculate connected components
            std::vector<int> currQueue;
            std::vector<int> nextQueue;
            int currSize = 0;
            currQueue.push_back(v);

            while (!currQueue.empty()) {
                for (int x : currQueue) {
                    // node already in a component, we can skip it
                    if (connectedComponent[x] != -1) {
                        continue;
                    }
                    // add x to the connected component
                    connectedComponent[x] = currComponent;
                    currSize++;

                    // add neighbors (that are not already explored) to queue
                    for (int y : g.getNeighbors(x)) {
                        nextQueue.push_back(y);
                    }
                }
                // update queues
                currQueue = nextQueue;
                nextQueue.clear();
            }
            componentSize.push_back(currSize);
            currComponent++;
        }
    }

    LOG_INFO("Found " << componentSize.size() << " components");
    ASSERT(componentSize.size() >= 1);
    ASSERT(std::accumulate(componentSize.begin(), componentSize.end(), 0) == n);
    return std::make_pair(connectedComponent, componentSize);
}

Graph GraphAlgo::getLargestComponent(Graph &unconnected) { return getLargestComponentWithMapping(unconnected).first; }

std::pair<Graph, std::map<NodeId, NodeId>> GraphAlgo::getLargestComponentWithMapping(Graph &unconnected) {
    auto cc = GraphAlgo::calculateComponentId(unconnected);

    std::vector<int> connectedComponent = cc.first;
    std::vector<int> componentSize = cc.second;

    // find largest component
    int largestComponent = -1;
    int largestSize = -1;
    for (int i = 0; i < componentSize.size(); i++) {
        if (componentSize[i] > largestSize) {
            largestSize = componentSize[i];
            largestComponent = i;
        }
    }

    // calculate the mapping of old to new nodeIds
    // also calculate number of edges in new graph
    std::unordered_map<int, int> nodeIdMapping;
    int currIdCounter = 0;
    int newEdges = 0;
    for (int v = 0; v < unconnected.getNumVertices(); v++) {
        if (connectedComponent[v] == largestComponent) {
            nodeIdMapping[v] = currIdCounter;
            currIdCounter++;

            // calculate number of new edges
            for (int u : unconnected.getNeighbors(v)) {
                if (connectedComponent[u] == largestComponent) {
                    newEdges++;
                }
            }
        }
    }
    newEdges /= 2;

    // add nodes and edges to new graph
    Graph connected;
    connected.setSize(largestSize, newEdges);
    for (int v = 0; v < unconnected.getNumVertices(); v++) {
        if (connectedComponent[v] == largestComponent) {
            for (int u : unconnected.getNeighbors(v)) {
                if (connectedComponent[u] == largestComponent) {
                    connected.addEdge(nodeIdMapping[u]);
                }
            }
            connected.nextNode();
        }
    }

    // calculate mapping von new id to old id
    std::map<NodeId, NodeId> newToOld;
    for (auto it = nodeIdMapping.begin(); it != nodeIdMapping.end(); it++) {
        newToOld[it->second] = it->first;
    }

    LOG_DEBUG("Calculated biggest component. Unconnected " << unconnected.getNumVertices() << ", Connected "
                                                           << connected.getNumVertices());
    return std::make_pair(connected, newToOld);
}

std::pair<Graph, std::vector<EdgeId>> GraphAlgo::coarsenGraph(Graph &g, const std::vector<NodeId> &clusterId) {
    ASSERT(Toolkit::noGapsInVector(clusterId));

    std::vector<EdgeId> resultMap(g.getNumEdges() * 2);

    std::vector<std::pair<NodeId, NodeId>> graphMap;
    for (NodeId v = 0; v < g.getNumVertices(); v++) {
        for (EdgeId e : g.getEdges(v)) {
            if (clusterId[v] != clusterId[g.getEdgeTarget(e)]) {
                std::pair<NodeId, NodeId> vu = std::make_pair(clusterId[v], clusterId[g.getEdgeTarget(e)]);
                graphMap.push_back(vu);
            }
        }
    }
    Graph result(graphMap);

    // NOTE: this is ugly. Can it be done in armotized O(n) without extra datastructures?
    std::map<std::pair<NodeId, NodeId>, EdgeId> edgeMapMap;
    for (NodeId v = 0; v < result.getNumVertices(); v++) {
        for (EdgeId e : result.getEdges(v)) {
            edgeMapMap[std::make_pair(v, result.getEdgeTarget(e))] = e;
        }
    }
    for (NodeId v = 0; v < g.getNumVertices(); v++) {
        for (EdgeId e : g.getEdges(v)) {
            if (clusterId[v] != clusterId[g.getEdgeTarget(e)]) {
                resultMap[e] = edgeMapMap[std::make_pair(clusterId[v], clusterId[g.getEdgeTarget(e)])];
            } else {
                resultMap[e] = -1;
            }
        }
    }

    return std::make_pair(result, resultMap);
}

std::vector<int> GraphAlgo::calculateShortestPaths(const Graph &g, NodeId origin) {
    const int N = g.getNumVertices();

    std::vector<int> distance(N, -1);

    // do BFS to calculate connected components
    std::vector<int> currQueue;
    std::vector<int> nextQueue;
    currQueue.push_back(origin);
    int currDist = 0;

    while (!currQueue.empty()) {
        for (int x : currQueue) {
            // node already visited, we can skip it
            if (distance[x] != -1) {
                continue;
            }
            distance[x] = currDist;
            // add neighbors (that are not already explored) to queue
            for (int y : g.getNeighbors(x)) {
                nextQueue.push_back(y);
            }
        }
        // update queues
        currDist++;
        currQueue = nextQueue;
        nextQueue.clear();
    }

    return distance;
}

std::vector<std::vector<int>> GraphAlgo::calculateAllPairShortestPaths(const Graph &g) {
    LOG_DEBUG("Calculating all pair shortest paths");
    const int N = g.getNumVertices();
    std::vector<std::vector<int>> allDistances(N);

    for (NodeId v = 0; v < N; v++) {
        allDistances[v] = calculateShortestPaths(g, v);
    }
    LOG_DEBUG("Finished calculating all pair shortest paths");
    return allDistances;
}
