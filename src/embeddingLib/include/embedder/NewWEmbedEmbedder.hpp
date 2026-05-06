#pragma once

#include "AdamOptimizer.hpp"
#include "EmbedderInterface.hpp"
#include "EmbedderOptions.hpp"
#include "VecList.hpp"
#include "WeightedIndex.hpp"

class NewWEmbedEmbedder : public EmbedderInterface {

    Graph graph;
    EmbedderOptions opts;
    std::shared_ptr<util::Timer> timer;

    uint32_t currentIteration = 0;
    uint32_t numRepForceCalculations = 0;
    VecList currentPositions;
    std::vector<double> currentWeights;
    std::vector<double> weightPrefixSum;
    std::vector<int32_t> sortedNodeIDs;
    std::vector<double> currentWeightParameters;

    AdamOptimizer posOptimizer;
    AdamOptimizer weightOptimizer;

    //TODO: Maybe better to use a parameter passed to a function or as a return value
    bool insignificantPosChange = false;

    [[nodiscard]] constexpr uint32_t graphSize() const;

    void computeWeightPrefixSum();
    void sortNodes();
    void attractionForce(NodeId v, NodeId u, VecList& force, VecBuffer<1>& buffer);
    void attractionWeightForce(NodeId v, NodeId u, std::vector<double>& weightParameterForce, VecBuffer<1>& buffer);
    void repellingForce(NodeId v, NodeId u, VecBuffer<1> forceBuffer, VecList &currentForce);
    void repellingWeightForce(NodeId v, NodeId u, VecBuffer<1> forceBuffer, std::vector<double> &weightParameterForce);

    void debug_dumpWeights() const;

    void updateIndex(std::vector<NodeId>& indexToGraphMap, WeightedIndex& currentWeightedIndex);
    std::vector<NodeId> getRepellingCandidatesForNode(NodeId v, VecBuffer<2> &buffer, WeightedIndex currentWeightedIndex, std::vector<NodeId>& indexToGraphMap) const;
    void calculateAllAttractingForces(VecList& force, std::vector<double>& weightParameterForce);
    void calculateAllRepellingForces(WeightedIndex currentWeightedIndex, std::vector<NodeId> &indexToGraphMap, VecList &currentForce, std::vector<double> &
                                     weightParameterForce);

    [[nodiscard]] std::vector<NodeId> sampleRandomNoise(int32_t numNodes) const;

    [[nodiscard]] std::vector<std::vector<double>> constructRandomCoordinates(uint32_t dimension) const;

    std::vector<double> rescaleWeights(double dimensionHint, double embeddingDimension,
                                       const std::vector<double> &weights);

    std::vector<double> constructDegreeWeights(const Graph &g);

    std::vector<double> constructUnitWeights(int N);

    public:
    NewWEmbedEmbedder(const Graph& g,
                      const EmbedderOptions &opts,
                      const std::shared_ptr<util::Timer> &timer_ptr = std::make_shared<util::Timer>())
                      : graph(g),
                        opts(opts),
                        timer(timer_ptr),
                        currentPositions(opts.embeddingDimension, g.getNumVertices()),
                        currentWeights(g.getNumVertices()),
                        weightPrefixSum(g.getNumVertices()),
                        sortedNodeIDs(g.getNumVertices()),
                        currentWeightParameters(g.getNumVertices()),
                        posOptimizer(opts.embeddingDimension, g.getNumVertices(), opts.learningRate, opts.coolingFactor, 0.9, 0.999, 1e-8),
                        weightOptimizer(opts.embeddingDimension, g.getNumVertices(), opts.weightLearningRate,opts.coolingFactor, 0.9, 0.999, 1e-8)
    {

        //TODO: Refactor all of the below called functions
        NewWEmbedEmbedder::setCoordinates(constructRandomCoordinates(opts.embeddingDimension));

        switch (opts.weightType) {
            case WeightType::Degree:
                NewWEmbedEmbedder::setWeights(NewWEmbedEmbedder::rescaleWeights(opts.dimensionHint, opts.embeddingDimension,
                                                                          NewWEmbedEmbedder::constructDegreeWeights(g)));
                break;
            case WeightType::Unit:
                NewWEmbedEmbedder::setWeights(NewWEmbedEmbedder::constructUnitWeights(graphSize()));
                break;
            default:
                LOG_ERROR("Weight type not supported");
        }
    }

    //TODO: Add Python Bindings?
    virtual ~NewWEmbedEmbedder() override = default;
    virtual void calculateStep() override;
    virtual bool isFinished() override;
    virtual void calculateEmbedding() override;
    virtual Graph getCurrentGraph() override;
    virtual std::vector<std::vector<double>> getCoordinates() override;
    virtual std::vector<double> getWeights() override;
    virtual std::vector<util::TimingResult> getTimings() override;
    virtual void setCoordinates(const std::vector<std::vector<double>> &coordinates) override;
    virtual void setWeights(const std::vector<double>& weights) override;
};
