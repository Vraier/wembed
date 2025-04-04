#pragma once

#include <memory>

#include "AdamOptimizer.hpp"
#include "EmbedderInterface.hpp"
#include "EmbedderOptions.hpp"
#include "GraphHierarchy.hpp"
#include "LabelPropagation.hpp"
#include "Timings.hpp"
#include "WEmbedEmbedder.hpp"
#include "WeightedIndex.hpp"

class LayeredEmbedder : public EmbedderInterface {
    using Timer = util::Timer;

   public:
    LayeredEmbedder(Graph &g, LabelPropagation &coarsener, EmbedderOptions opts)
        : timer(std::make_shared<Timer>()),
          options(opts),
          originalGraph(g),
          hierarchy(std::make_shared<GraphHierarchy>(g, coarsener)),
          currentLayer(hierarchy->getNumLayers() - 1),
          currentEmbedder(hierarchy->graphs[currentLayer], opts, timer) {};

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
    WEmbedEmbedder currentEmbedder;
};