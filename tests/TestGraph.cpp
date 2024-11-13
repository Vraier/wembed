#include <gtest/gtest.h>

#include "Graph.hpp"
#include "Macros.hpp"

TEST(Graph, Constructor) {
    Graph g;
    EXPECT_EQ(g.getNumVertices(), 0);
    EXPECT_EQ(g.getNumEdges(), 0);
}

TEST(Graph, EmptyConstruction) {
    std::map<int, std::set<int>> map;
    Graph g1(map);
    EXPECT_EQ(g1.getNumVertices(), 0);

    std::vector<std::pair<int, int>> set;
    Graph g2(set);
    EXPECT_EQ(g2.getNumVertices(), 0);
}

TEST(Graph, SelfLoop) {
    std::map<int, std::set<int>> map;
    map[0] = {0};
    Graph g(map);
    EXPECT_EQ(g.getNumVertices(), 1);
    EXPECT_EQ(g.getNumEdges(), 0);
    EXPECT_EQ(g.getNumNeighbors(0), 0);
}

TEST(Graph, SingleNode) {
    std::map<int, std::set<int>> map;
    map[0] = {};
    Graph g(map);
    EXPECT_EQ(g.getNumVertices(), 1);
    EXPECT_EQ(g.getNumEdges(), 0);
    EXPECT_EQ(g.getNumNeighbors(0), 0);
}

// test for constructing a graph with 3 nodes and 2 edges
TEST(Graph, ConstructGraph) {
    Graph g;
    g.setSize(3, 2);
    g.addEdge(1);
    g.nextNode();
    g.addEdge(0);
    g.addEdge(2);
    g.nextNode();
    g.addEdge(1);
    g.nextNode();

    EXPECT_EQ(g.getNumVertices(), 3);
    EXPECT_EQ(g.getNumEdges(), 2);

    EXPECT_EQ(g.getNumNeighbors(0), 1);
    EXPECT_EQ(g.getNumNeighbors(1), 2);
    EXPECT_EQ(g.getNumNeighbors(2), 1);
}

// test for reading in a graph from a map
TEST(Graph, ConstructGraphFromMap) {
    std::map<int, std::set<int>> map;
    map[0] = {1};
    map[1] = {0, 2};
    map[2] = {1};

    Graph g(map);

    EXPECT_EQ(g.getNumVertices(), 3);
    EXPECT_EQ(g.getNumEdges(), 2);

    EXPECT_EQ(g.getNumNeighbors(0), 1);
    EXPECT_EQ(g.getNumNeighbors(1), 2);
    EXPECT_EQ(g.getNumNeighbors(2), 1);

    EXPECT_EQ(g.getNeighbors(0)[0], 1);
    EXPECT_EQ(g.getNeighbors(1)[0], 0);
    EXPECT_EQ(g.getNeighbors(1)[1], 2);
}

// test for reading in a graph from a set of edges
TEST(Graph, ConstructGraphFromEdgeList) {
    std::vector<std::pair<NodeId, NodeId>> set;
    set.push_back({0, 1});
    set.push_back({1, 0});
    set.push_back({1, 2});
    set.push_back({2, 1});

    Graph g(set);

    EXPECT_EQ(g.getNumVertices(), 3);
    EXPECT_EQ(g.getNumEdges(), 2);

    EXPECT_EQ(g.getNumNeighbors(0), 1);
    EXPECT_EQ(g.getNumNeighbors(1), 2);
    EXPECT_EQ(g.getNumNeighbors(2), 1);

    EXPECT_EQ(g.getNeighbors(0)[0], 1);
    EXPECT_EQ(g.getNeighbors(1)[0], 0);
    EXPECT_EQ(g.getNeighbors(1)[1], 2);
}

// test for areNeighbors
TEST(Graph, AreNeighbors) {
    std::vector<std::pair<NodeId, NodeId>> set;
    set.push_back({0, 1});
    set.push_back({1, 0});
    set.push_back({1, 2});
    set.push_back({2, 1});
    set.push_back({0, 3});
    set.push_back({0, 3});
    set.push_back({3, 0});
    set.push_back({0, 4});
    set.push_back({4, 0});

    Graph g(set);

    EXPECT_TRUE(g.areNeighbors(0, 1));
    EXPECT_TRUE(g.areNeighbors(1, 0));
    EXPECT_TRUE(g.areNeighbors(1, 2));
    EXPECT_TRUE(g.areNeighbors(2, 1));
    EXPECT_TRUE(g.areNeighbors(0, 3));
    EXPECT_TRUE(g.areNeighbors(3, 0));
    EXPECT_TRUE(g.areNeighbors(0, 4));
    EXPECT_TRUE(g.areNeighbors(4, 0));

    EXPECT_FALSE(g.areNeighbors(0, 2));
    EXPECT_FALSE(g.areNeighbors(2, 0));
    EXPECT_FALSE(g.areNeighbors(1, 3));
    EXPECT_FALSE(g.areNeighbors(3, 1));
    EXPECT_FALSE(g.areNeighbors(1, 4));
    EXPECT_FALSE(g.areNeighbors(4, 1));
    EXPECT_FALSE(g.areNeighbors(2, 3));
    EXPECT_FALSE(g.areNeighbors(3, 2));
    EXPECT_FALSE(g.areNeighbors(2, 4));
    EXPECT_FALSE(g.areNeighbors(4, 2));
    EXPECT_FALSE(g.areNeighbors(3, 4));
    EXPECT_FALSE(g.areNeighbors(4, 3));
}
