#include "GraphHierarchy.hpp"

#include "GraphAlgorithms.hpp"

GraphHierarchy::GraphHierarchy(HierarchyOptions opts, const Graph& graph, const NodeParentPointer& parentPointer,
                               const std::vector<double>& initialWeights)
    : options(opts), NUMLAYERS(parentPointer.size()), buffer(opts.dimension) {
    LOG_INFO( "Starting to coarsen and embed graph hierarchy");
    const int DIMENSION = opts.dimension;
    ASSERT(DIMENSION > 0);
    ASSERT(parentPointer.size() > 0);
    ASSERT(opts.forceExponent > 0);

    // coarsen the graphs and convert them to embedded graphs
    std::vector<std::vector<EdgeId>> edgeParentPointers;
    Graph currGraph = graph;
    for (int i = 0; i < NUMLAYERS; i++) {
        graphs.push_back(EmbeddedGraph(DIMENSION, currGraph));

        if (i < NUMLAYERS - 1) {
            auto coarsened = GraphAlgo::coarsenGraph(currGraph, parentPointer[i]);
            currGraph = coarsened.first;
            edgeParentPointers.push_back(coarsened.second);
        }
    }
    // push back an empty pointer. last graph only has a single node and no edges
    edgeParentPointers.push_back(std::vector<EdgeId>());

    // reserve space for tree and sorted weights
    LOG_INFO( "Finished coarsening and embedding graph hierarchy");
    LOG_INFO( "Starting to construct parent tree");

    nodeLayers = std::vector<std::vector<NodeInformation>>(NUMLAYERS);
    edgeLayers = std::vector<std::vector<EdgeInformation>>(NUMLAYERS);
    for (int l = 0; l < NUMLAYERS; l++) {
        ASSERT(parentPointer[l].size() == graphs[l].getNumVertices());
        ASSERT(edgeParentPointers[l].size() == graphs[l].getNumEdges() * 2);

        nodeLayers[l] = std::vector<NodeInformation>(graphs[l].getNumVertices());
        edgeLayers[l] = std::vector<EdgeInformation>(graphs[l].getNumEdges() * 2);
    }

    // assign parent and child nodes
    for (int l = 0; l < NUMLAYERS - 1; l++)  // only do it for #layer -1
    {
        // assign pointers for nodes
        int nodeLayerSize = parentPointer[l].size();
        for (int i = 0; i < nodeLayerSize; i++) {
            nodeLayers[l][i].parentNode = parentPointer[l][i];             // assign parent
            nodeLayers[l + 1][parentPointer[l][i]].children.push_back(i);  // assign children
        }

        // assign pointers for edges
        int edgeLayerSize = edgeParentPointers[l].size();
        for (int i = 0; i < edgeLayerSize; i++) {
            edgeLayers[l][i].parentEdge = edgeParentPointers[l][i];  // assign parent
            if (edgeParentPointers[l][i] != -1) {
                // some edges don't exists in upper layers. we cant assign children to these
                edgeLayers[l + 1][edgeParentPointers[l][i]].children.push_back(i);  // assign children
            }
        }
    }

    sumUpValuesInNodes(initialWeights);
    sumUpValuesInEdges();

    LOG_INFO( "Finished constructing parent tree");
}

void GraphHierarchy::sumUpValuesInNodes(const std::vector<double>& initialWeights) {
    // count number of contained nodes and sum up scaled weights
    for (int l = 0; l < NUMLAYERS; l++) {
        int nodeLayerSize = nodeLayers[l].size();
        for (int i = 0; i < nodeLayerSize; i++) {
            if (l == 0) {
                // in the bottom layer every tree nodes contains exactly one actual node
                nodeLayers[l][i].nodeWeightSum = initialWeights[i];
                nodeLayers[l][i].totalContainedNodes = 1;
                nodeLayers[l][i].scaledNodeWeightSum =
                    std::pow(initialWeights[i], (double)options.forceExponent / (double)options.dimension);
                nodeLayers[l][i].inverseScaledNodeWeightSum = 1.0 / nodeLayers[l][i].scaledNodeWeightSum;
                graphs[l].setNodeWeight(i, nodeLayers[l][i].nodeWeightSum);

            } else {
                // in the other layers the sum of nodes get added up
                for (int c : nodeLayers[l][i].children) {
                    nodeLayers[l][i].nodeWeightSum += nodeLayers[l - 1][c].nodeWeightSum;
                    nodeLayers[l][i].totalContainedNodes += nodeLayers[l - 1][c].totalContainedNodes;
                    nodeLayers[l][i].scaledNodeWeightSum += nodeLayers[l - 1][c].scaledNodeWeightSum;
                    nodeLayers[l][i].inverseScaledNodeWeightSum += nodeLayers[l - 1][c].inverseScaledNodeWeightSum;
                }
                graphs[l].setNodeWeight(i, nodeLayers[l][i].nodeWeightSum);
            }
        }
    }
}

void GraphHierarchy::sumUpValuesInEdges() {
    // count number of contained edges
    for (int l = 0; l < NUMLAYERS; l++) {
        for (NodeId v = 0; v < graphs[l].getNumVertices(); v++) {
            for (EdgeId e : graphs[l].getEdges(v)) {
                if (l == 0) {
                    // in the bottom layer every edge contains exactly one actual edge
                    edgeLayers[l][e].totalContainedEdges = 1;
                    double wv = graphs[l].getNodeWeight(v);
                    double wu = graphs[l].getNodeWeight(graphs[l].getEdgeTarget(e));
                    edgeLayers[l][e].sumInverseEdgeWeight = 1.0 / std::pow(wv * wu, 1.0 / (double)options.dimension);
                } else {
                    // in the other layers the sum of edges get added up
                    for (int c : edgeLayers[l][e].children) {
                        edgeLayers[l][e].totalContainedEdges += edgeLayers[l - 1][c].totalContainedEdges;
                        edgeLayers[l][e].sumInverseEdgeWeight += edgeLayers[l - 1][c].sumInverseEdgeWeight;
                    }
                }
            }
        }
    }
}

int GraphHierarchy::getNumLayers() const { return NUMLAYERS; }

int GraphHierarchy::getLayerSize(int layer) const {
    ASSERT(layer < getNumLayers());

    return nodeLayers[layer].size();
}

double GraphHierarchy::getScaledWeightSum(int layer, NodeId node) const {
    ASSERT(layer < getNumLayers());
    ASSERT((0 <= node) && (node < getLayerSize(layer)),
           "node: " + std::to_string(node) + ", layerSize: " + std::to_string(getLayerSize(layer)));

    return nodeLayers[layer][node].scaledNodeWeightSum;
}

double GraphHierarchy::getInverseScaledWeightSum(int layer, NodeId node) const {
    ASSERT(layer < getNumLayers());
    ASSERT((0 <= node) && (node < getLayerSize(layer)),
           "node: " + std::to_string(node) + ", layerSize: " + std::to_string(getLayerSize(layer)));

    return nodeLayers[layer][node].inverseScaledNodeWeightSum;
}

double GraphHierarchy::getNodeWeight(int layer, NodeId node) const {
    ASSERT(layer < getNumLayers());
    ASSERT((0 <= node) && (node < getLayerSize(layer)),
           "node: " + std::to_string(node) + ", layerSize: " + std::to_string(getLayerSize(layer)));

    return graphs[layer].getNodeWeight(node);
}

double GraphHierarchy::getInverseEdgeWeightSum(int layer, EdgeId edge) const {
    return edgeLayers[layer][edge].sumInverseEdgeWeight;
}

int GraphHierarchy::getTotalContainedEdges(int layer, EdgeId edge) const {
    ASSERT(layer < getNumLayers());
    ASSERT((0 <= edge) && (edge < edgeLayers[layer].size()),
           "edge: " + std::to_string(edge) + ", layerSize: " + std::to_string(edgeLayers[layer].size()));
    return edgeLayers[layer][edge].totalContainedEdges;
}

int GraphHierarchy::getParent(int layer, NodeId node) const {
    ASSERT(layer < getNumLayers());
    ASSERT((0 <= node) && (node < getLayerSize(layer)),
           "node: " + std::to_string(node) + ", layerSize: " + std::to_string(getLayerSize(layer)));

    return nodeLayers[layer][node].parentNode;
}

CVecRef GraphHierarchy::getAveragePosition(int layer, NodeId node) const {
    ASSERT(layer < getNumLayers());
    ASSERT((0 <= node) && (node < getLayerSize(layer)),
           "node: " + std::to_string(node) + ", layerSize: " + std::to_string(getLayerSize(layer)));

    return graphs[layer].getPosition(node);
}

double GraphHierarchy::getNodeWeightSum(int layer, NodeId node) const {
    ASSERT(layer < getNumLayers());
    ASSERT((0 <= node) && (node < getLayerSize(layer)),
           "node: " + std::to_string(node) + ", layerSize: " + std::to_string(getLayerSize(layer)));

    return nodeLayers[layer][node].nodeWeightSum;
}

int GraphHierarchy::getTotalContainedNodes(int layer, int node) const {
    ASSERT(layer < getNumLayers());
    ASSERT((0 <= node) && (node < getLayerSize(layer)),
           "node: " + std::to_string(node) + ", layerSize: " + std::to_string(getLayerSize(layer)));

    return nodeLayers[layer][node].totalContainedNodes;
}

std::vector<NodeId> GraphHierarchy::getChildren(int layer, int node) const {
    ASSERT(layer < getNumLayers());
    ASSERT((0 <= node) && (node < getLayerSize(layer)),
           "node: " + std::to_string(node) + ", layerSize: " + std::to_string(getLayerSize(layer)));

    return nodeLayers[layer][node].children;
}

void GraphHierarchy::setPositionOfNode(int layer, int node, CVecRef position) {
    ASSERT(layer < getNumLayers());
    ASSERT((0 <= node) && (node < getLayerSize(layer)),
           "node: " + std::to_string(node) + ", layerSize: " + std::to_string(getLayerSize(layer)));

    TmpVec<0> tmpVec(buffer);

    CVecRef prevAvgPos = getAveragePosition(layer, node);
    const int nodesInBottomNode = getTotalContainedNodes(layer, node);

    // position difference. Nodes in upper layers have to also be moved by this
    // weighted difference
    tmpVec = position - prevAvgPos;

    // update all layers above this node
    int currNodeId = node;
    for (int l = layer; l < getNumLayers(); l++) {
        int nodesInParent = getTotalContainedNodes(l, currNodeId);
        // determines how much the position difference changes the position
        double changePercent = (double)nodesInBottomNode / (double)nodesInParent;
        graphs[l].coordinates[currNodeId] += changePercent * tmpVec;
        // find the next parent
        currNodeId = getParent(l, currNodeId);
    }
}

void GraphHierarchy::setNodeWeight(int layer, int node, double weight) {
    ASSERT(layer == 0);
    ASSERT((0 <= node) && (node < getLayerSize(layer)),
           "node: " + std::to_string(node) + ", layerSize: " + std::to_string(getLayerSize(layer)));

    // get the previous weights and the amount by which they must change
    double prevWeight = getNodeWeight(layer, node);
    double prevScaledWeight = getScaledWeightSum(layer, node);
    double prevInverseScaledWeight = getInverseScaledWeightSum(layer, node);

    double weightDiff = weight - prevWeight;
    double scaledWeightDiff =
        std::pow(weight, (double)options.forceExponent / (double)options.dimension) - prevScaledWeight;
    double inverseScaledWeightDiff =
        (1.0 / std::pow(weight, (double)options.forceExponent / (double)options.dimension)) - prevInverseScaledWeight;

    // update all layers above this node
    NodeId currNodeId = node;
    for (int l = layer; l < getNumLayers(); l++) {
        // determines the new weight
        double newWeight = getNodeWeight(l, currNodeId) + weightDiff;
        double newScaledWeight = getScaledWeightSum(l, currNodeId) + scaledWeightDiff;
        double newInverseScaledWeight = getInverseScaledWeightSum(l, currNodeId) + inverseScaledWeightDiff;

        // set the new weight
        graphs[l].setNodeWeight(currNodeId, newWeight);
        nodeLayers[l][currNodeId].scaledNodeWeightSum = newScaledWeight;
        nodeLayers[l][currNodeId].inverseScaledNodeWeightSum = newInverseScaledWeight;

        // find next parent
        currNodeId = getParent(l, currNodeId);
    }
}

void GraphHierarchy::applyPositionToChildren(int layer) {
    ASSERT((layer < getNumLayers()) && (layer > 0));

    const int lowerLayer = layer - 1;
    // find the parent for every node one layer down and set the position
    // accordingly
    for (NodeId c = 0; c < getLayerSize(lowerLayer); c++) {
        graphs[lowerLayer].coordinates[c] = graphs[layer].coordinates[getParent(lowerLayer, c)];
    }
}

std::string GraphHierarchy::toString() const {
    std::string result;
    for (int l = NUMLAYERS - 1; l >= 0; l--) {
        result += "Graph of layer " + std::to_string(l) + ":\n";
        result += graphs[l].toString();
    }

    result += "Partition tree:\n";
    for (int l = NUMLAYERS - 1; l >= 0; l--) {
        result += "Layer " + std::to_string(l) + ": ";
        for (int i = 0; i < getLayerSize(l); i++) {
            result += std::to_string(getParent(l, i)) + " ";
        }
        result += "\n";
    }
    return result;
}
