#pragma once

#include "EmbOptions.hpp"
#include "GraphHierarchy.hpp"
#include "VecList.hpp"
#include "WeightedApproxEmbedder.hpp"

class WeightedSpringEmbedder : public WeightedApproxEmbedder {
   public:
    WeightedSpringEmbedder(OptionValues opts, GraphHierarchy *h, int level) : options(opts),
                                                                              currIteration(0),
                                                                              hierarchy(h),
                                                                              LEVEL(level),
                                                                              oldPositions(options.dimension),
                                                                              currentForce(options.dimension),
                                                                              buffer(options.dimension){};
    virtual void initializeNewRun();
    virtual void calculateStep();
    virtual void calculateEmbedding();
    virtual bool isFinished();

    OptionValues options;

   private:
    /**
     * Updates the position of the nodes
     */
    void calculateForceStep();
    /**
     * Updates the weight of the nodes
     * Only sensible for single nodes and not cluster of nodes (?)
     */
    void calculateWeightStep();

    // adds the attracting force to every entry in currentForce
    void calculateAllAttractingForces();

    // adds the repelling force to every entry
    void calculateAllRepellingForces();

    /**
     * Maps a node id of the lowest graph level to the current level
     */
    NodeId lowestToCurrentLevel(NodeId v);

    /**
     * calculates all the repelling forces acting on this node.
     * also scales them accordingly
     */
    TmpCVec<0> sumRepulsionForce(int v);
    TmpCVec<1> repulsionForce(int v, int levelV, int u, int levelU);

    /**
     * determines the ideal edge length based on the weights of the nodes.
     * Nodes with higher weights should be further apart.
     */
    double idealEdgeLength(double wa, double wb);

    double getOptimalWeight(NodeId a);
    double getOptimalWeightUpperBound(NodeId a);
    double getOptimalWeightWeightedError(NodeId a);
    double getOptimalWeightExactError(NodeId a);
    double getOptimalWeightSampled(NodeId a);

    void addNeighborsToWeights(NodeId v, std::vector<std::pair<double, bool>> &weights);
    void addAllNodesToWeights(NodeId v, std::vector<std::pair<double, bool>> &weights);

    int getBestIndexAbsolute(NodeId v, const std::vector<std::pair<double, bool>> &weights) const;
    int getBestIndexRelative(NodeId v, const std::vector<std::pair<double, bool>> &weights) const;

    /**
     * Take a sorted list of weights and an index that determines what weights should be included.
     * Returns a threshold for which weights are to be included.
     */
    double getBestThresholdForIndex(int index, const std::vector<std::pair<double, bool>> &weights) const;

    int currIteration;
    int weightSteps;
    GraphHierarchy *hierarchy;
    std::vector<NodeId> lowestToCurrentMapping;
    const int LEVEL;

    // variables to measure performance
    double sumOfForces = 0;
    int numRepForceCalculations = 0;
    bool insignificantPosChange = false;
    bool insignificantWeightChange = false;

    VecList oldPositions;
    VecList currentForce;

    std::vector<double> oldWeights;
    std::vector<double> newWeights;

    VecBuffer<11> buffer;
};
