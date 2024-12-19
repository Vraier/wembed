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

std::pair<Graph, std::vector<EdgeId>> GraphAlgo::coarsenGraph(Graph &g, const std::vector<NodeId> &clusterId) {
    ASSERT(Toolkit::noGapsInVector(clusterId));

    std::vector<EdgeId> resultEdgeMap(g.getNumEdges() * 2);
    std::map<NodeId, std::set<NodeId>> graphMap;
    for (NodeId v = 0; v < g.getNumVertices(); v++) {
        // initialize the set for the node if it does not exist yet
        if(graphMap.find(clusterId[v]) == graphMap.end()){
            graphMap[clusterId[v]] = std::set<NodeId>();
        }
        for (EdgeId e : g.getEdges(v)) {
            if (clusterId[v] != clusterId[g.getEdgeTarget(e)]) {
                graphMap[clusterId[v]].insert(clusterId[g.getEdgeTarget(e)]);
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
                resultEdgeMap[e] = edgeMapMap[std::make_pair(clusterId[v], clusterId[g.getEdgeTarget(e)])];
            } else {
                resultEdgeMap[e] = -1;
            }
        }
    }

    return std::make_pair(result, resultEdgeMap);
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
