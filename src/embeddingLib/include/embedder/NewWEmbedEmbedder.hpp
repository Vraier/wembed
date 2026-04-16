#pragma once
#include <EmbedderOptions.hpp>
#include <VecList.hpp>

#include "EmbedderInterface.hpp"

class NewWEmbedEmbedder : public EmbedderInterface {

    Graph graph;
    EmbedderOptions opts;
    std::shared_ptr<util::Timer> timer;

    uint32_t currentIteration = 0;
    VecList currentPositions;
    std::vector<double> currentWeights;
    public:
    NewWEmbedEmbedder(const Graph& g,
                      const EmbedderOptions &opts,
                      const std::shared_ptr<util::Timer> &timer_ptr = std::make_shared<util::Timer>())
                      : graph(g),
                        opts(opts),
                        timer(timer_ptr),
                        currentPositions(opts.embeddingDimension, g.getNumVertices()),
                        currentWeights(g.getNumVertices())
    {
        //TODO: Initialize coordinates randomly and weights based on degree
    }

    virtual ~NewWEmbedEmbedder() override {};
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
