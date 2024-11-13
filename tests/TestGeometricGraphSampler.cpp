#include <gtest/gtest.h>

#include "GeometricGraphSampler.hpp"
#include "Macros.hpp"

// construct a random geometric graph and assert the average dergree is roughly 20
TEST(GeometricGraphSampler, RandomGeometricGraph) {
    GeometricGraphSampler sampler;
    Graph g = sampler.generateRandomGraph(4000); 

    // calculate the average degree
    double averageDegree = (double)g.getNumEdges()*2.0 / (double)g.getNumVertices();

    EXPECT_NEAR(averageDegree, 20, 1);
    EXPECT_NEAR(g.getNumVertices(), 4000, 200);
}

// test that the generated graph is connected
TEST(GeometricGraphSampler, RandomGeometricGraphConnected) {
    GeometricGraphSampler sampler;
    Graph g = sampler.generateRandomGraph(4000); 

    // check that the graph is connected
    std::vector<bool> visited(g.getNumVertices(), false);
    std::vector<NodeId> queue;
    queue.push_back(0);
    visited[0] = true;
    while (!queue.empty()) {
        NodeId v = queue.back();
        queue.pop_back();
        for (NodeId u : g.getNeighbors(v)) {
            if (!visited[u]) {
                visited[u] = true;
                queue.push_back(u);
            }
        }
    }

    for (NodeId v = 0; v < g.getNumVertices(); v++) {
        EXPECT_TRUE(visited[v]);
    }
}
