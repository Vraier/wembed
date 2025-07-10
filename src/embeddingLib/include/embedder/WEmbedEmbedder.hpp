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
          optimizer(opts.embeddingDimension, N, opts.learningRate, opts.coolingFactor, 0.9, 0.999, 1e-8),
          weightOptimizer(1, N, opts.weightLearningRate, opts.coolingFactor, 0.9, 0.999, 1e-8),
          currentweightedIndex(opts.indexType, opts.embeddingDimension),
          IndexToGraphMap(N),
          sortedNodeIds(N),
          currentForce(opts.embeddingDimension, N),
          currentPositions(opts.embeddingDimension, N),
          oldPositions(opts.embeddingDimension, N),
          currentWeights(N),
          currentWeightParameters(N),
          weightParameterForce(N),
          weightPrefixSum(N) {
        // Initialize coordinates randomly and weights based on degree
        setCoordinates(WEmbedEmbedder::constructRandomCoordinates(opts.embeddingDimension, N));

        switch (options.weightType) {
            case WeightType::Degree:
                setWeights(WEmbedEmbedder::rescaleWeights(opts.dimensionHint, opts.embeddingDimension,
                                                          WEmbedEmbedder::constructDegreeWeights(g)));
                break;
            case WeightType::Unit:
                setWeights(WEmbedEmbedder::constructUnitWeights(N));
                break;
            default:
                LOG_ERROR("Weight type not supported");
        }
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

    /**
     * Updates the weightForce vector
     */
    virtual void repulsionWeightForce(int v, int u, VecBuffer<1> &buffer);
    virtual void attractionWeightForce(int v, int u, VecBuffer<1> &buffer);
    virtual void calculateAllWeightPenalties();
    virtual void weightPenaltyForce(int v);  // tries to achive uniform weights

    // NOTE(JP) has race conditions because of randomness. Can also contain duplicates
    virtual std::vector<NodeId> sampleRandomNodes(int numNodes) const;
    void debug_dump_weights() const;  // appends the weights in the last iteration to a debug file

    /**
     * Methods on the spacial index
     */
    virtual void updateIndex();
    virtual std::vector<NodeId> getRepellingCandidatesForNode(NodeId v, VecBuffer<2> &buffer) const;

    std::shared_ptr<Timer> timer;
    EmbedderOptions options;
    Graph graph;
    int N;  // size of the graph

    // additional data structures
    AdamOptimizer optimizer;              // for positions
    AdamOptimizer weightOptimizer;        // for weights
    WeightedIndex currentweightedIndex;   // changes every iteration
    std::vector<NodeId> IndexToGraphMap;  // maps spacial indices to graph indices
    std::vector<int> sortedNodeIds;       // node ids sorted by weight

    int currentIteration = 0;
    bool insignificantPosChange = false;
    long long numRepForceCalculations = 0;  // number of repulsion force calculations in one step

    // current state of gradient calculation
    VecList currentForce;
    VecList currentPositions;
    VecList oldPositions;                         // needed to calulate the change in position
    std::vector<double> currentWeights;           // has to be kept in sync with the hidden parameters
    std::vector<double> currentWeightParameters;  // hidden parameter needed for optimization. w = log(1+e^x)
    std::vector<double> weightParameterForce;     // force acting on the hidden parameter
    std::vector<double> weightPrefixSum;  // starts at the weight of the first node and ends with the sum of all weights
};