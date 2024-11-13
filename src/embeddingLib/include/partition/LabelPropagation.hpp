#pragma once

#include "EmbeddedGraph.hpp"
#include "Partitioner.hpp"

using SingleLayerNodePointer = std::vector<NodeId>;
using NodeParentPointer = std::vector<SingleLayerNodePointer>;

class LabelPropagation : public Partitioner {
   public:
    LabelPropagation(PartitionerOptions ops, const Graph &g, const std::vector<double> &edgeWs);

    /**
     * Does lable propagation and coarsening until the graph is small enough
     */
    virtual NodeParentPointer coarsenAllLayers();

   private:
    PartitionerOptions options;
    Graph graph;
    std::vector<double> initialEdgeWeights;

    /**
     * Determines the clusterID for multiple rounds of label propagation.
     * Respects the maximum cluster size and maximum number of label propagation
     * rounds. Chooses the order in which nodes are processed according to the
     * given options.
     */
    SingleLayerNodePointer labelPropagation(const Graph &currG, const std::vector<double> &edgeWs);

    /**
     * Will be executed when the graph does not get coarsened enough.
     * Ignores the cluster size limit but ensures that the hierarchy has
     * logarithmic hight.
     */
    SingleLayerNodePointer aggressivePropagation(const Graph &currG, const std::vector<double> &edgeWs,
                                                        const SingleLayerNodePointer &currParents);

    SingleLayerNodePointer calculateLabelPropagationOrder(const Graph &currG);

    /**
     * Reassigns the cluster IDs such that there are no gaps between IDs
     * e.g. [1,3,1,6,6,5,5] -> [0,1,0,2,2,3,3]
     */
    SingleLayerNodePointer compactClusterIds(const SingleLayerNodePointer &clusterIds);

    /**
     * Sums up the edge weights based on the edgeMap
     */
    static std::vector<double> calculateNewEdgeWeights(const std::vector<double> &oldWeights,
                                                       const std::vector<EdgeId> &edgeMap);
};
