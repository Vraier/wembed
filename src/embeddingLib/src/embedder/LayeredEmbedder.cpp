#include "LayeredEmbedder.hpp"

#include "Macros.hpp"

void LayeredEmbedder::calculateStep() {
    currentIteration++;
    if (currentEmbedder->isFinished()) {
        expandPositions();
    }
    currentEmbedder->calculateStep();
}

bool LayeredEmbedder::isFinished() { return (currentLayer == 0) && currentEmbedder->isFinished(); }

void LayeredEmbedder::calculateEmbedding() {
    LOG_INFO("Calculating embedding...");
    timer->startTiming("embedding_all", "Embedding");
    currentIteration = 0;
    while (!isFinished()) {
        calculateStep();
    }
    timer->stopTiming("embedding_all");
    LOG_INFO("Finished calculating embedding in iteration " << currentIteration);
}

void LayeredEmbedder::setCoordinates(const std::vector<std::vector<float>>& coordinates) {
    LOG_WARNING("Setting coordinates for layered embedder has no effect");
    unused(coordinates);
    return;
}

void LayeredEmbedder::setWeights(const std::vector<float>& weights) {
    LOG_WARNING("Setting weights for layered embedder has no effect");
    unused(weights);
    return;
}

std::vector<std::vector<float>> LayeredEmbedder::getCoordinates() { return currentEmbedder->getCoordinates(); }

std::vector<float> LayeredEmbedder::getWeights() { return currentEmbedder->getWeights(); }

std::vector<util::TimingResult> LayeredEmbedder::getTimings() { return timer->getHierarchicalTimingResults(); }

Graph LayeredEmbedder::getCurrentGraph() { return hierarchy->graphs[currentLayer]; }

void LayeredEmbedder::expandPositions() {
    LOG_INFO("Expanding positions to layer " << currentLayer - 1 << " in iteration " << currentIteration);
    timer->startTiming("expanding", "Expanding Positions");

    VecBuffer<1> buffer(opts.embeddingDimension);
    TmpVec<0> tmpVec(buffer);

    int newN = hierarchy->graphs[currentLayer - 1].getNumVertices();
    int oldN = hierarchy->graphs[currentLayer].getNumVertices();
    std::vector<std::vector<float>> oldPostions = currentEmbedder->getCoordinates();
    std::vector<std::vector<float>> newPositions(newN, std::vector<float>(opts.embeddingDimension, 0.0));
    ASSERT(oldN == oldPostions.size(), "Old positions size mismatch: " << oldN << " vs " << oldPostions.size());

    // calculate new weights
    std::vector<float> newWeights;
    if (opts.weightType == WeightType::Degree) {
        newWeights =
            WembedEmbedder::rescaleWeights(opts.dimensionHint, static_cast<float>(opts.embeddingDimension),
                                           WembedEmbedder::constructDegreeWeights(hierarchy->graphs[currentLayer - 1]));
    } else if (opts.weightType == WeightType::Unit) {
        newWeights = WembedEmbedder::constructUnitWeights(newN);
    } else {
        LOG_ERROR("Weight type not supported");
    }

    // calculate new positions
    float geometricStretch = Toolkit::myPowf(static_cast<float>(newN) / static_cast<float>(oldN), 1.0f / static_cast<float>(opts.embeddingDimension));
    geometricStretch *= opts.expansionStretch;
    for (int v = 0; v < newN; v++) {
        int parent = hierarchy->nodeLayers[currentLayer - 1][v].parentNode;
        ASSERT(parent < oldN, "Parent node " << parent << " is out of bounds " << oldN);
        float numSiblings = static_cast<float>(hierarchy->nodeLayers[currentLayer][parent].totalContainedNodes);

        tmpVec.setToRandomUnitVector();
        float sphere_size = Toolkit::myPowf(numSiblings, 1.0f / static_cast<float>(opts.embeddingDimension));
        tmpVec *= sphere_size;
        for (int d = 0; d < opts.embeddingDimension; d++) {
            newPositions[v][d] = geometricStretch * oldPostions[parent][d] + static_cast<float>(tmpVec[d]);
        }
    }

    currentLayer--;
    currentEmbedder = std::make_unique<WembedEmbedder>(hierarchy->graphs[currentLayer], opts, timer, false);
    currentEmbedder->setCoordinates(newPositions);
    currentEmbedder->setWeights(newWeights);

    timer->stopTiming("expanding");
}
