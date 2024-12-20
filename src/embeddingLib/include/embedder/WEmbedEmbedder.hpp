#pragma once

#include "AdamOptimizer.hpp"
#include "EmbedderInterface.hpp"
#include "EmbedderOptions.hpp"
#include "Graph.hpp"
#include "Timings.hpp"
#include "VecList.hpp"
#include "WeightedRTree.hpp"

class WEmbedEmbedder : public EmbedderInterface {
    using Timer = util::Timer;

   public:
    WEmbedEmbedder(Graph &g, EmbedderOptions opts)
        : options(opts),
          graph(g),
          optimizer(opts.embeddingDimension, g.getNumVertices(), opts.speed, opts.coolingFactor, 0.9, 0.999, 10e-8),
          currentRTree(opts.embeddingDimension),
          currentForce(opts.embeddingDimension, g.getNumVertices()),
          currentPositions(opts.embeddingDimension, g.getNumVertices()),
          oldPositions(opts.embeddingDimension, g.getNumVertices()),
          currentWeights(g.getNumVertices()) {
        // Initialize coordinates randomly and weights based on degree
        setCoordinates(WEmbedEmbedder::constructRandomCoordinates(opts.embeddingDimension, g.getNumVertices()));
        setWeights(WEmbedEmbedder::rescaleWeights(opts.dimensionHint, opts.embeddingDimension,
                                                  WEmbedEmbedder::constructDegreeWeights(g)));
        optimizer.reset();
    };

    virtual ~WEmbedEmbedder() {};

    virtual void calculateStep();
    virtual bool isFinished();
    virtual void calculateEmbedding();

    virtual Graph getCurrentGraph();
    virtual std::vector<std::vector<double>> getCoordinates();
    virtual std::vector<double> getWeights();

    virtual void setCoordinates(const std::vector<std::vector<double>> &coordinates);
    virtual void setWeights(const std::vector<double> &weights);

    std::vector<util::TimingResult> getTimings() const;

    // Functions for calculating initial layouts
    static std::vector<std::vector<double>> constructRandomCoordinates(int dimension, int numVertices);
    static std::vector<double> constructDegreeWeights(const Graph &g);
    static std::vector<double> constructUnitWeights(int N);  // TODO: use the unit weights
    static std::vector<double> rescaleWeights(int dimensionHint, int embeddingDimension,
                                              const std::vector<double> &weights);

   private:
    /**
     * Updates the currentForce vector
     */
    virtual void calculateAllAttractingForces();
    virtual void calculateAllRepellingForces();
    virtual void repulstionForce(int v, int u, VecBuffer<1> &buffer);
    virtual void attractionForce(int v, int u, VecBuffer<1> &buffer);

    /**
     * R-Tree queries
     */
    virtual void updateRTree();
    virtual std::vector<NodeId> getRepellingCandidatesForNode(NodeId v, VecBuffer<2> &buffer) const;

    Timer timer;
    EmbedderOptions options;
    Graph graph;

    // additional data structures
    AdamOptimizer optimizer;
    WeightedRTree currentRTree;  // changes every iteration

    int currentIteration = 0;
    bool insignificantPosChange = false;

    // current state of gradient calculation
    VecList currentForce;
    VecList currentPositions;
    VecList oldPositions;
    std::vector<double> currentWeights;  // currently not changed during gradient descent

    //
};