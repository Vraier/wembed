#pragma once

#include "EmbedderOptions.hpp"
#include "GraphHierarchy.hpp"
#include "VecList.hpp"
#include "WeightedApproxEmbedder.hpp"

class AbstractSimpleEmbedder : public WeightedApproxEmbedder {
   public:
    AbstractSimpleEmbedder(EmbedderOptions opts, GraphHierarchy *h, int level)
        : options(opts),
          buffer(options.embeddingDimension),
          hierarchy(h),
          LEVEL(level),
          graph(hierarchy->graphs[LEVEL]),
          oldPositions(options.embeddingDimension),
          currentForce(options.embeddingDimension) {};

    virtual void initializeNewRun();
    virtual void calculateStep();
    virtual bool isFinished();

    virtual void calculateEmbedding();

    EmbedderOptions options;

   protected:
    VecBuffer<20> buffer;
    static constexpr int ATTR_BUFFER = 0;
    static constexpr int REP_BUFFER = 1;
    static constexpr int FORCE_STEP_BUFFER = 2;
    static constexpr int ALL_ATTR_BUFFER = 3;
    static constexpr int ALL_REP_BUFFER = 4;
    static constexpr int REP_AUX_BUFFER = 5;  // to find closest node for repelling forces

    /**
     * Updates the position of the nodes
     */
    virtual void calculateForceStep();

    /**
     * Updates the weight of the nodes
     * Only sensible for single nodes and not cluster of nodes (?)
     */
    virtual void calculateWeightStep();

    // adds the attracting force to every entry in currentForce
    virtual void calculateAllAttractingForces();

    // adds the repelling force to every entry
    virtual void calculateAllRepellingForces();

    // finds the first layer that is smaller than the approx factor and calculates the repelling forces with all nodes
    // from there
    void calculateAllRepellingForcesCoarse();

    // uses a priority list to find the nearest clusters in the highest resolution
    void calculateAllRepellingForcesPriority();

    /**
     * Attraction force that node in target excerpts on v
     */
    virtual TmpCVec<ATTR_BUFFER> attractionForce(int v, int level, EdgeId target) = 0;
    /**
     * repulsion force the node cluster u excerpts on v
     */
    virtual TmpCVec<REP_BUFFER> repulsionForce(int v, int levelV, int u, int levelU) = 0;

    GraphHierarchy *hierarchy;
    const int LEVEL;

    EmbeddedGraph &graph;

    // variables to measure performance
    int numIterations = 0;
    int numForceSteps = 0;
    int numWeightSteps = 0;
    int numRepForceCalculations = 0;

    bool insignificantPosChange = false;
    bool insignificantWeightChange = false;

    VecList oldPositions;
    VecList currentForce;
};
