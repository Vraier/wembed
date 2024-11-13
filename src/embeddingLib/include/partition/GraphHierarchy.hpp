#pragma once

#include "EmbeddedGraph.hpp"

using SingleLayerNodePointer = std::vector<NodeId>;
using NodeParentPointer = std::vector<SingleLayerNodePointer>;

// TODO:
// unify the node weights of embedded graph and graph heirarchy
// update the weights of nodes and edges accordingly, when weights get changed

struct HierarchyOptions {
    int dimension = -1;
    double forceExponent = -1.0;  // needed to scale weights accordingly when summing up values
};

struct NodeInformation {
    // pointer to next and previous layer
    int parentNode = -1;
    std::vector<int> children;  // This is kind of needed for calculating approx rep forces

    // information about all contained nodes
    int totalContainedNodes;
    double nodeWeightSum;               // sum of the node weights
    double scaledNodeWeightSum;         // x = w^(f/d): needed for repulsive force calculation
    double inverseScaledNodeWeightSum;  // y = 1/x: needed for sigmoid rep. force calculation: sum 1/wa^1/d
};

struct EdgeInformation {
    // pointer to next and previous layer
    int parentEdge = 1;
    std::vector<int> children;

    // information about weights and counts
    int totalContainedEdges;      // counts the number of edges between clusters
    double sumInverseEdgeWeight;  // 1 / (wa * wb)^(1/d)
};

/**
 * Stores information about the graph hierarchy
 * Caches values that help to speed up the calculation of the forces
 */
class GraphHierarchy {
   public:
    GraphHierarchy(HierarchyOptions opts, const Graph& graph, const NodeParentPointer& parentPointer,
                   const std::vector<double>& initialWeights);
    ~GraphHierarchy(){};

    int getNumLayers() const;
    int getLayerSize(int layer) const;

    // numbers concerning specific nodes and edges
    NodeId getParent(int layer, NodeId node) const;
    CVecRef getAveragePosition(int layer, NodeId node) const;


    double getNodeWeightSum(int layer, NodeId node) const;
    /**
     * Returns sum w^(q/d)
     * Needed for calculating repelling forces
     */
    double getScaledWeightSum(int layer, NodeId node) const;
    /**
     * Returns sum 1/w^(q/d)
     * Needed for calculating repelling forces, for sigmoid embedder
     */
    double getInverseScaledWeightSum(int layer, NodeId node) const;
    /**
     * Returns the exact weight of a node.
     * For Nodes in higher layers this is the sum of the contained nodes.
     * Does only make sense for the lowest layer. Needed for attracting forces.
     */
    double getNodeWeight(int layer, NodeId node) const;

    /**
     * Returns sum 1 / (wa * wb)^(1/d)
     */
    double getInverseEdgeWeightSum(int layer, EdgeId edge) const;

    /**
     * Returns the number of edges between the two cluster of nodes
     */
    int getTotalContainedEdges(int layer, EdgeId edge) const;
    int getTotalContainedNodes(int layer, int node) const;

    /**
     * Returns a list of all nodes contained in the cluster
     */
    std::vector<NodeId> getChildren(int layer, int node) const;

    /**
     * Updates the position of the given node.
     * Also changes average position of all corresponding nodes in the upper layers.
     *
     * Only updates the position for nodes of higher levels but not lower levels.
     * The positions of lower levels become invalid if this function is called
     * Lower levels become valid by calling applyPositionToChildren
     */
    void setPositionOfNode(int layer, int node, CVecRef position);

    /**
     * Updates the weight of a node and all layers above the node.
     * Must only be called on the lowest layer.
     */
    void setNodeWeight(int layer, int node, double weight);

    /**
     * For all nodes x in this layer, sets the positions of the children from node x to the position of x
     */
    void applyPositionToChildren(int layer);

    std::string toString() const;

    // TODO(h): this should be private
    std::vector<EmbeddedGraph> graphs;

   private:
    // used during hierarchy construction
    void sumUpValuesInNodes(const std::vector<double>& initialWeights);
    void sumUpValuesInEdges();

    const HierarchyOptions options;
    const int NUMLAYERS;

    std::vector<std::vector<NodeInformation>> nodeLayers;
    std::vector<std::vector<EdgeInformation>> edgeLayers;

    VecBuffer<10> buffer;
};
