#pragma once

#include "EmbeddedGraph.hpp"
#include "GraphHierarchy.hpp"

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

/**
 * Interface for all partitioning algorithms.
 */
class Partitioner {
   public:
    virtual ~Partitioner(){};

    virtual NodeParentPointer coarsenAllLayers() = 0;
};