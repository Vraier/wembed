#pragma once

#include <memory>

#include "AdamOptimizer.hpp"
#include "EmbedderInterface.hpp"
#include "EmbedderOptions.hpp"
#include "GraphHierarchy.hpp"
#include "LabelPropagation.hpp"
#include "Timings.hpp"
#include "WEmbedEmbedder.hpp"
#include "WeightedRTree.hpp"

class SingleLayerEmbedder {
    using Timer = util::Timer;

   public:
    SingleLayerEmbedder(std::shared_ptr<GraphHierarchy> hierarchy, int layer, EmbedderOptions opts,
                        std::shared_ptr<Timer> timer_ptr)
        : timer(timer_ptr),
          options(opts),
          LAYER(layer),
          graph(hierarchy->graphs[LAYER]),
          N(hierarchy->getLayerSize(LAYER)),
          optimizer(opts.embeddingDimension, N, opts.speed, opts.coolingFactor, 0.9, 0.999, 10e-8),
          currentRTree(opts.embeddingDimension),
          hierarchy(hierarchy),
          currentForce(opts.embeddingDimension, N),
          currentPositions(opts.embeddingDimension, N),
          oldPositions(opts.embeddingDimension, N),
          currentWeights(N) {
        optimizer.reset();
    };

    virtual void calculateStep();
    virtual bool isFinished();
    virtual void calculateEmbedding();

    virtual std::vector<std::vector<double>> getCoordinates();
    virtual std::vector<double> getWeights();

    virtual void setCoordinates(const std::vector<std::vector<double>> &coordinates);
    virtual void setWeights(const std::vector<double> &weights);

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

    std::shared_ptr<Timer> timer;
    EmbedderOptions options;
    int LAYER;  // layer in the hierachy
    Graph graph;
    int N;  // size of the graph

    // additional data structures
    AdamOptimizer optimizer;
    WeightedRTree currentRTree;      // changes every iteration
    std::vector<int> sortedNodeIds;  // node ids sorted by weight

    bool insignificantPosChange = false;
    int currentIteration = 0;

    // current state of gradient calculation
    std::shared_ptr<GraphHierarchy> hierarchy;
    VecList currentForce;
    VecList currentPositions;
    VecList oldPositions;
    std::vector<double> currentWeights;  // currently not changed during gradient descent
};

class LayeredEmbedder : public EmbedderInterface {
    using Timer = util::Timer;

   public:
    LayeredEmbedder(Graph &g, LabelPropagation &coarsener, EmbedderOptions opts)
        : timer(std::make_shared<Timer>()),
          options(opts),
          originalGraph(g),
          hierarchy(std::make_shared<GraphHierarchy>(g, coarsener)),
          currentLayer(hierarchy->getNumLayers() - 1),
          currentEmbedder(hierarchy, currentLayer, opts, timer) {};

    virtual void calculateStep();
    virtual bool isFinished();
    virtual void calculateEmbedding();

    virtual void setCoordinates(const std::vector<std::vector<double>> &coordinates);
    virtual void setWeights(const std::vector<double> &weights);

    virtual std::vector<std::vector<double>> getCoordinates();
    virtual std::vector<double> getWeights();
    virtual std::vector<util::TimingResult> getTimings();
    virtual Graph getCurrentGraph();

   private:
    std::shared_ptr<Timer> timer;

    // decreases the layer and initializes a new embedder
    virtual void expandPositions();

    EmbedderOptions options;
    Graph originalGraph;
    std::shared_ptr<GraphHierarchy> hierarchy;

    int currentIteration = 0;
    int currentLayer;
    bool insignificantPosChange = false;

    // stores positions and weights of all graphs in the hierarchy
    SingleLayerEmbedder currentEmbedder;
};