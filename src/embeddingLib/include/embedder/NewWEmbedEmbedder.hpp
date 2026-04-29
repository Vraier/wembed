#pragma once

#include <execution>

#include "EmbedderInterface.hpp"
#include "EmbedderOptions.hpp"
#include "VecList.hpp"
#include "WeightedIndex.hpp"

class NewWEmbedEmbedder : public EmbedderInterface {

    Graph graph;
    EmbedderOptions opts;
    std::shared_ptr<util::Timer> timer;

    uint32_t currentIteration = 0;
    VecList currentPositions;
    std::vector<double> currentWeights;
    std::vector<double> weightPrefixSum;
    std::vector<int32_t> sortedNodeIDs;
    std::vector<double> currentWeightParameters;

    //TODO: Maybe better to use a parameter passed to a function or as a return value
    bool insignificantPosChange = false;

    [[nodiscard]] constexpr uint32_t graphSize() const;

    void computeWeightPrefixSum();
    void sortNodes();
    void attractionForce(const NodeId v, const NodeId u, VecList& force, VecBuffer<1>& buffer);
    void attractionWeightForce(const NodeId v, const NodeId u, std::vector<double>& weightParameterForce, VecBuffer<1>& buffer);

    void debug_dumpWeights() const;

    void updateIndex(std::vector<NodeId>& indexToGraphMap, WeightedIndex& currentWeightedIndex);
    void calculateAllAttractingForces(VecList& force, std::vector<double>& weightParameterForce);
    void calculateAllRepellingForces();

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
                        currentWeightParameters(g.getNumVertices())
    {
        //TODO: Initialize coordinates randomly and weights based on degree
    }

    virtual ~NewWEmbedEmbedder() override {}
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
