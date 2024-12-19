#include <gtest/gtest.h>

#include "Graph.hpp"
#include "GraphAlgorithms.hpp"
#include "Macros.hpp"

// test for calculating the connected components of a graph
TEST(GraphAlgorithm, ConnectedComponents) {
    std::map<int, std::set<int>> map;
    map[0] = {1};
    map[1] = {0, 2};
    map[2] = {1};
    map[3] = {4};
    map[4] = {3};

    Graph g(map);

    auto result = GraphAlgo::calculateComponentId(g);
    auto componentId = result.first;
    auto componentSize = result.second;

    EXPECT_EQ(componentId[0], 0);
    EXPECT_EQ(componentId[1], 0);
    EXPECT_EQ(componentId[2], 0);
    EXPECT_EQ(componentId[3], 1);
    EXPECT_EQ(componentId[4], 1);

    EXPECT_EQ(componentSize[0], 3);
    EXPECT_EQ(componentSize[1], 2);
}

// test for coarsening a graph
TEST(GraphAlgorithm, Coarsening) {
    std::map<int, std::set<int>> map;
    map[0] = {1, 2};
    map[1] = {0, 2};
    map[2] = {0, 1, 3, 4};
    map[3] = {2};
    map[4] = {2, 5, 6};
    map[5] = {4, 6};
    map[6] = {5, 4};

    Graph g(map);

    std::vector<NodeId> clusterId = {0, 0, 1, 2, 3, 3, 3};  // coarsen 0,1 and 4,5,6

    auto result = GraphAlgo::coarsenGraph(g, clusterId);
    Graph coarsened = result.first;
    std::vector<EdgeId> edgeMap = result.second;

    EXPECT_EQ(coarsened.getNumVertices(), 4);
    EXPECT_EQ(coarsened.getNumEdges(), 3);

    EXPECT_EQ(coarsened.getNumNeighbors(0), 1);
    EXPECT_EQ(coarsened.getNumNeighbors(1), 3);
    EXPECT_EQ(coarsened.getNumNeighbors(2), 1);
    EXPECT_EQ(coarsened.getNumNeighbors(3), 1);

    for (NodeId v = 0; v < g.getNumVertices(); v++) {
        for (EdgeId e : g.getEdges(v)) {
            NodeId u = g.getEdgeTarget(e);
            if (clusterId[v] != clusterId[u]) {
                ASSERT_NE(edgeMap[e], -1);
                ASSERT_LE(edgeMap[e], coarsened.getNumEdges() * 2);

                EXPECT_EQ(coarsened.getEdgeTarget(edgeMap[e]), clusterId[u]) << g.toString() + " " + coarsened.toString();
                EXPECT_TRUE(coarsened.areNeighbors(clusterId[v], clusterId[u]));
            } else {
                EXPECT_EQ(edgeMap[e], -1);
                EXPECT_FALSE(coarsened.areNeighbors(clusterId[v], clusterId[u]));
            }
        }
    }
}

// calculate shortest paths
TEST(GraphAlgorithm, ShortestPaths) {
    std::map<int, std::set<int>> map;
    map[0] = {1, 2};
    map[1] = {0, 2};
    map[2] = {0, 1, 3, 4};
    map[3] = {2};
    map[4] = {2, 5, 6};
    map[5] = {4, 6};
    map[6] = {5, 4};

    Graph g(map);

    std::vector<int> shortestPaths = GraphAlgo::calculateShortestPaths(g, 0);

    EXPECT_EQ(shortestPaths[0], 0);
    EXPECT_EQ(shortestPaths[1], 1);
    EXPECT_EQ(shortestPaths[2], 1);
    EXPECT_EQ(shortestPaths[3], 2);
    EXPECT_EQ(shortestPaths[4], 2);
    EXPECT_EQ(shortestPaths[5], 3);
    EXPECT_EQ(shortestPaths[6], 3);

    std::vector<int> shortestPaths2 = GraphAlgo::calculateShortestPaths(g, 2);

    EXPECT_EQ(shortestPaths2[0], 1);
    EXPECT_EQ(shortestPaths2[1], 1);
    EXPECT_EQ(shortestPaths2[2], 0);
    EXPECT_EQ(shortestPaths2[3], 1);
    EXPECT_EQ(shortestPaths2[4], 1);
    EXPECT_EQ(shortestPaths2[5], 2);
    EXPECT_EQ(shortestPaths2[6], 2);
}