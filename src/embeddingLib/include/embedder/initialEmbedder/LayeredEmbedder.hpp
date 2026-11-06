#pragma once

#include <memory>

#include "InitialEmbedderInterface.hpp"
#include "../EmbedderOptions.hpp"
#include "GraphHierarchy.hpp"
#include "LabelPropagation.hpp"
#include "Timings.hpp"
#include "WembedEmbedder.hpp"

class LayeredEmbedder : public InitialEmbedderInterface {
    using Timer = util::Timer;

   public:
    LayeredEmbedder(const Graph& g, LabelPropagation& coarsener, EmbedderOptions opts,
                    std::shared_ptr<Timer> timer = std::make_shared<Timer>());
    ~LayeredEmbedder() override = default;

    std::vector<std::vector<double>> getInitialEmbedding() override;

    EmbeddingProgress getProgress();
    std::vector<util::TimingResult> getTimings();

   private:
    void calculateStep();
    bool isFinished();
    void calculateEmbedding();

    std::shared_ptr<Timer> timer;

    // decreases the layer and initializes a new embedder
    void expandPositions();

    std::shared_ptr<GraphHierarchy> hierarchy;

    int currentLayer;
    // non-trivial layers only, their mean iteration count estimates the current layer
    int finishedLayers = 0;
    int finishedLayerIterations = 0;

    // stores positions and weights of all graphs in the hierarchy
    std::unique_ptr<WembedEmbedder> currentEmbedder;
};