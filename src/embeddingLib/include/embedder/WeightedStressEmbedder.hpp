#pragma once

#include "GraphHierarchy.hpp"
#include "EmbOptions.hpp"
#include "VecList.hpp"
#include "WeightedApproxEmbedder.hpp"

class WeightedStressEmbedder : public WeightedApproxEmbedder {
   public:
    WeightedStressEmbedder(OptionValues opts, GraphHierarchy *h, int level) : options(opts),
                                                                              hierarchy(h),
                                                                              LEVEL(level),
                                                                              oldPositions(options.dimension),
                                                                              newPositions(options.dimension),
                                                                              buffer(options.dimension){};
    virtual void initializeNewRun();
    virtual void calculateStep();
    virtual void calculateEmbedding();
    virtual bool isFinished();

    OptionValues options;

   private:
    // adds the atracting force to every entry in currentForce
    // also calculates the rohs array
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
     * Nodes with higher weigths should be further appart.
     */
    double idealEdgeLenght(double wa, double wb);

    GraphHierarchy *hierarchy;
    std::vector<NodeId> lowestToCurrentMapping;
    const int LEVEL;

    // keep state of which iteration is currently done
    int roundCounter = 0;
    int iterationCounter = 0;
    double currentAlpha;

    // variables to measure performance
    double sumOfFoces = 0;
    int numRepForceCalculations = 0;
    bool insignificantChange = false;

    VecList oldPositions;
    VecList newPositions;
    std::vector<double> rohs;

    VecBuffer<10> buffer;
};