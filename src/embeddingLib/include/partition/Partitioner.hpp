#pragma once

#include "Graph.hpp"

/**
 * Configures the partitioner needed for the hierarchical embedding
 * can determine how coarse/fine the partitioning is and the size of the hierarchy
 */
struct PartitionerOptions {
    int partitionType = 0;
    int maxIterations = 20;
    int maxClusterSize = 6;
    int finalGraphSize = 10;
    int orderType = 0;
    int numHierarchies = 1;
};

using ParentPointerTree = std::vector<std::vector<NodeId>>;

/**
 * Interface for all partitioning algorithms.
 */
class Partitioner {
   public:
    virtual ~Partitioner(){};

    virtual ParentPointerTree coarsenAllLayers() = 0;
};