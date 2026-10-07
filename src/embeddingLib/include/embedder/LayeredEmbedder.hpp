#pragma once

#include <memory>

#include "EmbedderInterface.hpp"
#include "EmbedderOptions.hpp"
#include "GraphHierarchy.hpp"
#include "LabelPropagation.hpp"
#include "Timings.hpp"
#include "WembedEmbedder.hpp"

class LayeredEmbedder : public EmbedderInterface {
    //TODO: remove redundant variables
    using Timer = util::Timer;

   public:
    LayeredEmbedder(const Graph &g, LabelPropagation &coarsener, EmbedderOptions opts)
        : EmbedderInterface(g, opts),
          timer(std::make_shared<Timer>()),
          hierarchy(std::make_shared<GraphHierarchy>(g, coarsener)),
          currentLayer(hierarchy->getNumLayers() - 1) {
        currentEmbedder = std::make_unique<WembedEmbedder>(hierarchy->graphs[currentLayer], opts, timer);
    }

    virtual void calculateStep();
    virtual bool isFinished();
    EmbeddingProgress getProgress() override;
    virtual void calculateEmbedding();

    virtual void setCoordinates(const std::vector<std::vector<flt_t>> &coordinates);
    virtual void setWeights(const std::vector<flt_t> &weights);

    virtual std::vector<std::vector<flt_t>> getCoordinates();
    virtual std::vector<flt_t> getWeights();
    virtual std::vector<util::TimingResult> getTimings();
    virtual Graph getCurrentGraph();

    int getNumVertices() const override { return currentEmbedder->getNumVertices(); }
    int getEmbeddingDimension() const override { return currentEmbedder->getEmbeddingDimension(); }
    void copyCoordinatesTo(flt_t* out) const override { currentEmbedder->copyCoordinatesTo(out); }
    EmbeddingLoss getLoss() const override { return currentEmbedder->getLoss(); }
    flt_t getCurrentLearningRate() const override { return currentEmbedder->getCurrentLearningRate(); }
    flt_t getLastRelDisplacement() const override { return currentEmbedder->getLastRelDisplacement(); }
    flt_t getLastRelLossImprovement() const override { return currentEmbedder->getLastRelLossImprovement(); }

   private:
    std::shared_ptr<Timer> timer;

    // decreases the layer and initializes a new embedder
    virtual void expandPositions();

    std::shared_ptr<GraphHierarchy> hierarchy;

    int currentLayer;
    // non-trivial layers only, their mean iteration count estimates the current layer
    int finishedLayers = 0;
    int finishedLayerIterations = 0;

    // stores positions and weights of all graphs in the hierarchy
    std::unique_ptr<WembedEmbedder> currentEmbedder;
};