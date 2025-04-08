#pragma once

#include "AdamOptimizer.hpp"
#include "EmbedderInterface.hpp"
#include "EmbedderOptions.hpp"
#include "Graph.hpp"
#include "Timings.hpp"
#include "VecList.hpp"
#include "WeightedIndex.hpp"

class WEmbedEmbedder : public EmbedderInterface {
    using Timer = util::Timer;

   public:
    WEmbedEmbedder(Graph &g, EmbedderOptions opts, std::shared_ptr<Timer> timer_ptr = std::make_shared<Timer>())
        : timer(timer_ptr),
          options(opts),
          graph(g),
          N(g.getNumVertices()),
          optimizer(opts.embeddingDimension, N, opts.speed, opts.coolingFactor, 0.9, 0.999, 1e-8),
          currentRTree(opts.embeddingDimension),
          sortedNodeIds(N),
          currentForce(opts.embeddingDimension, N),
          currentPositions(opts.embeddingDimension, N),
          oldPositions(opts.embeddingDimension, N),
          currentWeights(N) {
        // Initialize coordinates randomly and weights based on degree
        setCoordinates(WEmbedEmbedder::constructRandomCoordinates(opts.embeddingDimension, N));
        if (opts.weightType == WeightType::Degree) {
            setWeights(WEmbedEmbedder::rescaleWeights(opts.dimensionHint, opts.embeddingDimension,
                                                      WEmbedEmbedder::constructDegreeWeights(g)));
        } else if (opts.weightType == WeightType::Unit) {
            setWeights(WEmbedEmbedder::constructUnitWeights(N));
        } else {
            LOG_ERROR("Weight type not supported");
        }
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
    std::vector<util::TimingResult> getTimings();

    // Functions for calculating initial layouts
    static std::vector<std::vector<double>> constructRandomCoordinates(int dimension, int numVertices);
    static std::vector<double> constructDegreeWeights(const Graph &g);
    static std::vector<double> constructUnitWeights(int N);
    static std::vector<double> rescaleWeights(double dimensionHint, double embeddingDimension,
                                              const std::vector<double> &weights);

   private:
    /**
     * Updates the currentForce vector
     */
    virtual void calculateAllAttractingForces();
    virtual void calculateAllRepellingForces();
    virtual void repulstionForce(int v, int u, VecBuffer<1> &buffer);
    virtual void attractionForce(int v, int u, VecBuffer<1> &buffer);
    virtual std::vector<NodeId> sampleRandomNodes(
        int numNodes) const;  // NOTE(JP) has race conditions because of randomness

    /**
     * R-Tree queries
     */
    virtual void updateRTree();
    virtual std::vector<NodeId> getRepellingCandidatesForNode(NodeId v, VecBuffer<2> &buffer) const;

    std::shared_ptr<Timer> timer;
    EmbedderOptions options;
    Graph graph;
    int N;  // size of the graph

    // additional data structures
    AdamOptimizer optimizer;
    WeightedIndex currentRTree;      // changes every iteration
    std::vector<int> sortedNodeIds;  // node ids sorted by weight

    int currentIteration = 0;
    bool insignificantPosChange = false;

    // current state of gradient calculation
    VecList currentForce;
    VecList currentPositions;
    VecList oldPositions;
    std::vector<double> currentWeights;   // currently not changed during gradient descent
    std::vector<double> weightPrefixSum;  // starts at the weight of the first node and ends with the sum of all weights
};