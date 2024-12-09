#include "LabelPropagation.hpp"

#include "GraphAlgorithms.hpp"
#include "Macros.hpp"
#include "Rand.hpp"
#include "Toolkit.hpp"

LabelPropagation::LabelPropagation(PartitionerOptions ops, const Graph& g, const std::vector<double>& edgeWs)
    : options(ops), graph(g), initialEdgeWeights(edgeWs) {
    ASSERT(edgeWs.size() == g.getNumEdges() * 2, "Number of edge weights does not match number of edges");
}

ParentPointerTree LabelPropagation::coarsenAllLayers() {
    // hold information about all coarsened graphs
    ParentPointerTree parentPointers;
    std::vector<Graph> coarsenedGraphs;
    std::vector<std::vector<double>> edgeWeights;
    coarsenedGraphs.push_back(graph);
    edgeWeights.push_back(this->initialEdgeWeights);

    // determines by what factor the number of nodes were reduces by coarsening
    double shrinkFactor = 0;  // always do a normal label propagation at the start

    // coarsen the graph until we achieved the desired size
    while (coarsenedGraphs.back().getNumVertices() > options.finalGraphSize) {
        SingleLayerNodePointer nextMapping;

        if (shrinkFactor < 0.5) {
            // normal label propagation
            nextMapping = LabelPropagation::labelPropagation(coarsenedGraphs.back(), edgeWeights.back());
        } else {
            LOG_DEBUG("Graph only shrank by " << shrinkFactor << ". Performing aggressive coarsening.");
            nextMapping = LabelPropagation::aggressivePropagation(coarsenedGraphs.back(), edgeWeights.back(),
                                                                  parentPointers.back());
        }

        auto newGraph = GraphAlgo::coarsenGraph(coarsenedGraphs.back(), nextMapping);
        ASSERT(GraphAlgo::isConnected(newGraph.first), "Coarsened graph is not connected");
        parentPointers.push_back(nextMapping);
        coarsenedGraphs.push_back(newGraph.first);
        edgeWeights.push_back(calculateNewEdgeWeights(edgeWeights.back(), newGraph.second));
        shrinkFactor = (double)coarsenedGraphs[coarsenedGraphs.size() - 1].getNumVertices() /
                       (double)coarsenedGraphs[coarsenedGraphs.size() - 2].getNumVertices();
    }

    // create the last mappings (technical details)
    int numNodesLastGraph = coarsenedGraphs.back().getNumVertices();
    std::vector<NodeId> prevLastMapping(numNodesLastGraph, 0);  // all nodes get mapped into a single node
    std::vector<NodeId> lastMapping(1, -1);                     // end of graph hierarchy

    parentPointers.push_back(prevLastMapping);
    parentPointers.push_back(lastMapping);

    return parentPointers;
}

SingleLayerNodePointer LabelPropagation::labelPropagation(const Graph& currG, const std::vector<double>& edgeWs) {
    ASSERT(edgeWs.size() == currG.getNumEdges() * 2, "Number of edge weights does not match number of edges");

    const int NUM_ITERATIONS = options.maxIterations;
    const int CLUSTER_SIZE = options.maxClusterSize;
    const int N = currG.getNumVertices();

    std::vector<NodeId> nodeOrder = calculateLabelPropagationOrder(currG);
    std::vector<NodeId> clusterId(N);
    std::vector<double> edgeSum(N, 0);
    std::vector<int> clusterSize(N, 0);

    // every node starts in its own cluster
    for (int i = 0; i < N; i++) {
        clusterId[i] = i;
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        for (int v = 0; v < N; v++) {
            NodeId currentNode = nodeOrder[v];

            // find the cluster that this nodes has the most edges to
            // first: sum up the weights to the cluster of the edges
            for (EdgeId e : currG.getEdges(currentNode)) {
                edgeSum[clusterId[currG.getEdgeTarget(e)]] += edgeWs[e];
            }

            // second: determine cluster with most edges
            NodeId largestCluster = clusterId[currentNode];
            NodeId originalCluster = clusterId[currentNode];
            double maxWeight = 0;
            for (EdgeContent e : currG.getEdgeContents(currentNode)) {
                NodeId potentialCluster = clusterId[e.neighbour];
                if (edgeSum[potentialCluster] > maxWeight &&
                    ((clusterSize[potentialCluster] + 1) <= CLUSTER_SIZE || potentialCluster == originalCluster)) {
                    // new cluster must not be too big
                    // don't move the node if it already is in the best cluster
                    maxWeight = edgeSum[clusterId[e.neighbour]];
                    largestCluster = clusterId[e.neighbour];
                }
                // also reset the weight sum to zero
                edgeSum[clusterId[e.neighbour]] = 0;
            }

            // update cluster arrays
            clusterSize[largestCluster] += 1;
            clusterSize[originalCluster] -= 1;
            clusterId[currentNode] = largestCluster;
        }
    }

    return compactClusterIds(clusterId);
}

SingleLayerNodePointer LabelPropagation::aggressivePropagation(const Graph& currG, const std::vector<double>& edgeWs,
                                                               const SingleLayerNodePointer& currParents) {
    ASSERT(edgeWs.size() == currG.getNumEdges() * 2,
           "Number of edge weights " << edgeWs.size() << " does not match number of edges " << currG.getNumEdges() * 2);

    const int N = currG.getNumVertices();

    std::vector<int> numChildren(N, 0);
    std::vector<NodeId> clusterId(N, -1);
    std::vector<double> edgeSum(N, 0);

    // count how many nodes were clustered in the last step
    for (NodeId c = 0; c < currParents.size(); c++) {
        numChildren[currParents[c]] += 1;  // the parent of c received a node to its cluster
    }

    // now cluster the nodes that have only one child
    for (NodeId v = 0; v < N; v++) {
        if (numChildren[v] > 1) {
            // node stays in its own cluster
            clusterId[v] = v;
            continue;
        }

        // v is now a node that was not merged in the last step
        // v will not have an edge to another node that was not merge as they would have been contracted in the previous
        // step otherwise

        // determine the cluster to which v has the most edge weight
        for (EdgeId e : currG.getEdges(v)) {
            edgeSum[currG.getEdgeTarget(e)] += edgeWs[e];
        }
        NodeId largestCluster = -1;
        double maxWeight = -1;
        ASSERT(currG.getEdges(v).size() > 0, "Node " << v << " has no edges");
        for (EdgeId e : currG.getEdges(v)) {
            NodeId target = currG.getEdgeTarget(e);
            if (edgeSum[target] > maxWeight) {
                maxWeight = edgeSum[target];
                largestCluster = target;
            }
            // also reset the weight sum to zero
            edgeSum[target] = 0;
        }

        // update cluster arrays
        clusterId[v] = largestCluster;
    }
    return compactClusterIds(clusterId);
}

std::vector<NodeId> LabelPropagation::calculateLabelPropagationOrder(const Graph& currG) {
    std::vector<NodeId> result;

    switch (options.orderType) {
        case 0:
            for (NodeId v = 0; v < currG.getNumVertices(); v++) {
                result.push_back(v);
            }
            std::sort(result.begin(), result.end(), [&](const NodeId& lhs, const NodeId& rhs) -> bool {
                return (currG.getNumNeighbors(lhs) < currG.getNumNeighbors(rhs));
            });
            break;
        case 1:
            result = Rand::randomPermutation(currG.getNumVertices());
            break;
        default:
            LOG_ERROR("Unknown order type for label propagation");
            break;
    }
    return result;
}

SingleLayerNodePointer LabelPropagation::compactClusterIds(const SingleLayerNodePointer& clusterIds) {
    const int N = clusterIds.size();
    SingleLayerNodePointer idMap(N, -1);
    SingleLayerNodePointer compacted(N);
    int currId = 0;
    for (int v = 0; v < N; v++) {
        ASSERT(clusterIds[v] >= 0 && clusterIds[v] < N,
               "Cluster id " << clusterIds[v] << " out of bounds [0," << N << "]");
        if (idMap[clusterIds[v]] == -1) {
            // cluster does not have a mapping yet
            idMap[clusterIds[v]] = currId;
            currId++;
        }
        compacted[v] = idMap[clusterIds[v]];
    }

    ASSERT(Toolkit::noGapsInVector(compacted), "There are gaps in the cluster ids");
    return compacted;
}

std::vector<double> LabelPropagation::calculateNewEdgeWeights(const std::vector<double>& oldWeights,
                                                              const std::vector<EdgeId>& edgeMap) {
    ASSERT(Toolkit::noGapsInVector(edgeMap), "There are gaps in the edge map");
    ASSERT(Toolkit::findMinMax(edgeMap).first >= -1);

    EdgeId numNewEdges = Toolkit::findMinMax(edgeMap).second + 1;
    std::vector<double> newWeights(numNewEdges, 0);

    for (EdgeId e = 0; e < oldWeights.size(); e++) {
        if (edgeMap[e] == -1) {
            continue;
        }
        newWeights[edgeMap[e]] += oldWeights[e];
    }

    return newWeights;
}
