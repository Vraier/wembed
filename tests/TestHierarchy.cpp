#include <gtest/gtest.h>

#include <map>
#include <set>
#include <vector>

#include "Graph.hpp"
#include "GraphHierarchy.hpp"
#include "LabelPropagation.hpp"
#include "Rand.hpp"

namespace {

// connected grid, big enough to force several hierarchy layers
Graph makeGridGraph(int side) {
    std::vector<std::pair<int, int>> edges;
    auto id = [side](int x, int y) { return x * side + y; };
    for (int x = 0; x < side; x++) {
        for (int y = 0; y < side; y++) {
            if (x + 1 < side) edges.emplace_back(id(x, y), id(x + 1, y));
            if (y + 1 < side) edges.emplace_back(id(x, y), id(x, y + 1));
        }
    }
    return Graph(edges);
}

}  // namespace

TEST(Hierarchy, ContainedStatistics) {
    Rand::setSeed(1234);
    Graph g = makeGridGraph(20);
    std::vector<double> edgeWeights(g.getNumEdges() * 2, 1.0);
    LabelPropagation coarsener(PartitionerOptions{}, g, edgeWeights);
    GraphHierarchy hierarchy(g, coarsener);

    const int N = g.getNumVertices();
    const int numLayers = hierarchy.getNumLayers();
    ASSERT_GE(numLayers, 2);

    for (NodeId v = 0; v < N; v++) {
        EXPECT_EQ(hierarchy.nodeLayers[0][v].totalContainedNodes, 1);
        EXPECT_DOUBLE_EQ(hierarchy.nodeLayers[0][v].nodeWeightSum, g.getNumNeighbors(v));
    }
    for (size_t e = 0; e < hierarchy.edgeLayers[0].size(); e++) {
        EXPECT_EQ(hierarchy.edgeLayers[0][e].totalContainedEdges, 1);
    }

    // parents aggregate exactly their children
    for (int l = 0; l + 1 < numLayers; l++) {
        for (NodeId p = 0; p < hierarchy.getLayerSize(l + 1); p++) {
            const NodeInformation& info = hierarchy.nodeLayers[l + 1][p];
            EXPECT_GE(info.children.size(), 1u);
            int containedSum = 0;
            double weightSum = 0;
            for (NodeId c : info.children) {
                containedSum += hierarchy.nodeLayers[l][c].totalContainedNodes;
                weightSum += hierarchy.nodeLayers[l][c].nodeWeightSum;
            }
            EXPECT_EQ(info.totalContainedNodes, containedSum);
            EXPECT_DOUBLE_EQ(info.nodeWeightSum, weightSum);
        }
        for (size_t e = 0; e < hierarchy.edgeLayers[l + 1].size(); e++) {
            const EdgeInformation& info = hierarchy.edgeLayers[l + 1][e];
            EXPECT_GE(info.children.size(), 1u);
            int edgeSum = 0;
            for (int c : info.children) {
                edgeSum += hierarchy.edgeLayers[l][c].totalContainedEdges;
            }
            EXPECT_EQ(info.totalContainedEdges, edgeSum);
        }
    }

    // every layer partitions the leaves
    for (int l = 0; l < numLayers; l++) {
        int containedTotal = 0;
        double weightTotal = 0;
        for (NodeId v = 0; v < hierarchy.getLayerSize(l); v++) {
            containedTotal += hierarchy.nodeLayers[l][v].totalContainedNodes;
            weightTotal += hierarchy.nodeLayers[l][v].nodeWeightSum;
        }
        EXPECT_EQ(containedTotal, N);
        EXPECT_DOUBLE_EQ(weightTotal, 2.0 * g.getNumEdges());
    }
}

TEST(Hierarchy, IsolatedVerticesTerminate) {
    Rand::setSeed(1234);
    std::map<int, std::set<int>> adj;
    auto addEdge = [&](int u, int v) {
        adj[u].insert(v);
        adj[v].insert(u);
    };
    // six pairs of 6-cliques joined by a single edge
    int n = 0;
    for (int pair = 0; pair < 6; pair++) {
        for (int side = 0; side < 2; side++) {
            for (int i = 0; i < 6; i++) {
                for (int j = i + 1; j < 6; j++) addEdge(n + 6 * side + i, n + 6 * side + j);
            }
        }
        addEdge(n, n + 6);
        n += 12;
    }
    for (int i = 0; i < 100; i++) adj[n++];
    Graph g(adj);
    ASSERT_EQ(g.getNumVertices(), n);

    std::vector<double> edgeWeights(g.getNumEdges() * 2, 1.0);
    LabelPropagation coarsener(PartitionerOptions{}, g, edgeWeights);
    ParentPointerTree parents = coarsener.coarsenAllLayers();
    EXPECT_EQ(parents.back().size(), 1u);
}
