#include <gtest/gtest.h>

#include "Graph.hpp"
#include "GraphHierarchy.hpp"
#include "Macros.hpp"
#include "Toolkit.hpp"

void checkNodeAveragePositionConsistency(const GraphHierarchy &hierarchy, int layer, NodeId node) {
    VecList averagePositions(hierarchy.graphs[layer].getDimension());
    averagePositions.setSize(1, 0);

    for (NodeId child : hierarchy.getChildren(layer, node)) {
        averagePositions[0] +=
            hierarchy.getAveragePosition(layer - 1, child) * hierarchy.getTotalContainedNodes(layer - 1, child);
    }

    averagePositions[0] /= hierarchy.getTotalContainedNodes(layer, node);

    for (int i = 0; i < hierarchy.graphs[layer].getDimension(); i++) {
        EXPECT_NEAR(averagePositions[0][i], hierarchy.getAveragePosition(layer, node)[i], 1e-6)
            << "Layer: " << layer << " Node: " << node;
    }
}

void checkLayerAveragePositionConsistency(const GraphHierarchy &hierarchy, int layer) {
    for (NodeId node = 0; node < hierarchy.getLayerSize(layer); node++) {
        checkNodeAveragePositionConsistency(hierarchy, layer, node);
    }
}

void checkUpperLayerAveragePositionConsistency(const GraphHierarchy &hierarchy, int layer) {
    for (int l = layer; l < hierarchy.getNumLayers(); l++) {
        checkLayerAveragePositionConsistency(hierarchy, l);
    }
}

std::pair<GraphHierarchy, std::vector<double>> createPathHierarchy(int k, int dimension, double forceExp) {
    // create a path of length 2^k
    int n = (1 << k);
    std::vector<std::pair<NodeId, NodeId>> edges;
    for (NodeId v = 0; v < n; v++) {
        if (v < n - 1) {
            edges.push_back({v, v + 1});
            edges.push_back({v + 1, v});
        }
    }
    Graph g(edges);

    // calculate parent pointers
    std::vector<std::vector<NodeId>> nodeParentPointers;
    for (int layer = 0; layer < k; layer++) {
        std::vector<NodeId> tmp;
        for (int i = 0; i < (1 << (k - layer)); i++) {
            tmp.push_back(i / 2);
        }
        nodeParentPointers.push_back(tmp);
    }
    nodeParentPointers.push_back({-1});

    // calculate inital weights
    std::vector<double> initialWeights(n);
    for (int i = 0; i < n; i++) {
        initialWeights[i] = i;
    }

    GraphHierarchy hierarchy({dimension, forceExp}, g, nodeParentPointers, initialWeights);

    return {hierarchy, initialWeights};
}

TEST(Hierarchy, PathGraph) {
    const int dimension = 7;
    const double forceExponent = 1.45;
    const int k = 5;
    const int NUM_RANDOM_POS = 1000;

    auto res = createPathHierarchy(k, dimension, forceExponent);
    GraphHierarchy hierarchy = res.first;
    std::vector<double> initialWeights = res.second;

    // test node weights
    for (int layer = 0; layer < hierarchy.getNumLayers(); layer++) {
        for (NodeId node = 0; node < hierarchy.getLayerSize(layer); node++) {
            double weightSum = 0;
            double scaledNodeWeightSum = 0;
            double inverseScaledNodeWeightSum = 0;
            for (NodeId contained = node * (1 << layer); contained < (node + 1) * (1 << layer); contained++) {
                weightSum += initialWeights[contained];
                scaledNodeWeightSum += std::pow(initialWeights[contained], forceExponent / dimension);
                inverseScaledNodeWeightSum += 1.0 / std::pow(initialWeights[contained], forceExponent / dimension);
            }
            EXPECT_EQ(hierarchy.getTotalContainedNodes(layer, node), (1 << layer));
            EXPECT_DOUBLE_EQ(hierarchy.getNodeWeightSum(layer, node), weightSum);
            EXPECT_DOUBLE_EQ(hierarchy.getScaledWeightSum(layer, node), scaledNodeWeightSum);
            EXPECT_DOUBLE_EQ(hierarchy.getInverseScaledWeightSum(layer, node), inverseScaledNodeWeightSum);
        }
    }

    // test edge weights
    for (int layer = 0; layer < hierarchy.getNumLayers(); layer++) {
        for (NodeId node = 0; node < hierarchy.getLayerSize(layer) - 1; node++) {
            for (EdgeId edge : hierarchy.graphs[layer].getEdges(node)) {
                NodeId target = hierarchy.graphs[layer].getEdgeTarget(edge);

                NodeId lowNodeId = std::min(node, target);
                unused(lowNodeId);
                NodeId highNodeId = std::max(node, target);

                NodeId rightNodeInLowestLayer = highNodeId * (1 << layer);
                NodeId leftNodeInLowestLayer = rightNodeInLowestLayer - 1;
                double wa = initialWeights[rightNodeInLowestLayer];
                double wb = initialWeights[leftNodeInLowestLayer];
                double InverseEdgeWeight = 1.0 / std::pow(wa * wb, 1.0 / dimension);

                // only one edge between clusters in path graph
                EXPECT_EQ(hierarchy.getTotalContainedEdges(layer, edge), 1);
                EXPECT_DOUBLE_EQ(hierarchy.getInverseEdgeWeightSum(layer, edge), InverseEdgeWeight);
            }
        }
    }

    // test average position
    // go layer from top to bottom.
    // set random position at random node in current layer
    // check all upper layers for consistency
    for (int layer = hierarchy.getNumLayers() - 2; layer >= 0; layer--) {
        hierarchy.applyPositionToChildren(layer + 1);
        checkUpperLayerAveragePositionConsistency(hierarchy, layer + 1);

        for (int i = 0; i < NUM_RANDOM_POS; i++) {
            NodeId node = rand() % hierarchy.getLayerSize(layer);
            VecList randomPosition(dimension);
            randomPosition.setSize(1, 0);
            randomPosition[0].setToRandomUnitVector();
            randomPosition[0] *= 10000;
            hierarchy.setPositionOfNode(layer, node, randomPosition[0]);
            checkUpperLayerAveragePositionConsistency(hierarchy, layer + 1);
        }
    }
}