#pragma once

#include <execution>

#include "EmbedderInterface.hpp"
#include "EmbedderOptions.hpp"
#include "VecList.hpp"

class NewWEmbedEmbedder : public EmbedderInterface {

    Graph graph;
    EmbedderOptions opts;
    std::shared_ptr<util::Timer> timer;

    uint32_t currentIteration = 0;
    VecList currentPositions;
    std::vector<double> currentWeights;
    std::vector<double> weightPrefixSum;
    std::vector<uint32_t> sortedNodeIDs;

    [[nodiscard]] uint32_t graphSize() const {
        return this->graph.getNumVertices();
    }

    void computeWeightPrefixSum() {
        //TODO: parallel?
        weightPrefixSum[0] = currentWeights[0];
        for (size_t i = 1; i < currentWeights.size(); i++) {
            weightPrefixSum[i] = currentWeights[i] + weightPrefixSum[i - 1];
        }
    }

    void sortNodes() {
        std::iota(sortedNodeIDs.begin(), sortedNodeIDs.end(), 0);
        std::sort(std::execution::par_unseq, sortedNodeIDs.begin(), sortedNodeIDs.end(),
                  [this](const int a , const int b) -> bool {return this->currentWeights[a] > this->currentWeights[b];});
    }

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
                        sortedNodeIDs(g.getNumVertices())
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
