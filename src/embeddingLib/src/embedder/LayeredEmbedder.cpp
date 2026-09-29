#include "LayeredEmbedder.hpp"

#include <algorithm>
#include <stdexcept>

#include "Macros.hpp"
#include "ProgressEstimate.hpp"

void LayeredEmbedder::calculateStep() {
    if (currentEmbedder->isFinished()) {
        expandPositions();
    }
    currentEmbedder->calculateStep();
}

bool LayeredEmbedder::isFinished() { return (currentLayer == 0) && currentEmbedder->isFinished(); }

EmbeddingProgress LayeredEmbedder::getProgress() {
    EmbeddingProgress progress = currentEmbedder->getProgress();
    progress.layer = currentLayer;
    progress.numLayers = hierarchy->getNumLayers();
    if (finishedLayers > 0) {
        progress.expectedIterations = std::min(finishedLayerIterations / finishedLayers, opts.maxIterations);
    }
    // a coarse layer can't estimate the finer layers still to come
    progress.etaSeconds =
        currentLayer == 0
            ? progressEstimate::remainingLayerSeconds(progress.iteration, progress.expectedIterations,
                                                      progress.layerSeconds)
            : -1.0;
    return progress;
}

void LayeredEmbedder::calculateEmbedding() {
    timer->startTiming("embedding_all", "Embedding");
    while (!isFinished()) {
        calculateStep();
    }
    timer->stopTiming("embedding_all");
}

void LayeredEmbedder::setCoordinates(const std::vector<std::vector<double>>&) {
    throw std::logic_error("setCoordinates is not supported by the layered embedder");
}

void LayeredEmbedder::setWeights(const std::vector<double>&) {
    throw std::logic_error("setWeights is not supported by the layered embedder");
}

std::vector<std::vector<double>> LayeredEmbedder::getCoordinates() { return currentEmbedder->getCoordinates(); }

std::vector<double> LayeredEmbedder::getWeights() { return currentEmbedder->getWeights(); }

std::vector<util::TimingResult> LayeredEmbedder::getTimings() { return timer->getHierarchicalTimingResults(); }

Graph LayeredEmbedder::getCurrentGraph() { return hierarchy->graphs[currentLayer]; }

void LayeredEmbedder::expandPositions() {
    const EmbeddingProgress finished = currentEmbedder->getProgress();
    if (finished.numVertices > 1) {
        finishedLayers++;
        finishedLayerIterations += finished.iteration;
    }

    timer->startTiming("expanding", "Expanding Positions");

    VecBuffer<1> buffer(opts.embeddingDimension);
    TmpVec<0> tmpVec(buffer);

    int newN = hierarchy->graphs[currentLayer - 1].getNumVertices();
    int oldN = hierarchy->graphs[currentLayer].getNumVertices();
    std::vector<std::vector<double>> oldPostions = currentEmbedder->getCoordinates();
    std::vector<std::vector<double>> newPositions(newN, std::vector<double>(opts.embeddingDimension, 0.0));
    ASSERT(oldN == oldPostions.size(), "Old positions size mismatch: " << oldN << " vs " << oldPostions.size());

    // calculate new weights
    std::vector<double> newWeights;
    if (opts.weightType == WeightType::Degree) {
        newWeights =
            WembedEmbedder::rescaleWeights(opts.dimensionHint, opts.embeddingDimension,
                                           WembedEmbedder::constructDegreeWeights(hierarchy->graphs[currentLayer - 1]));
    } else if (opts.weightType == WeightType::Unit) {
        newWeights = WembedEmbedder::constructUnitWeights(newN);
    } else {
        throw std::invalid_argument("weight type not supported");
    }

    // calculate new positions
    double geometricStretch = Toolkit::myPow((double)newN / (double)oldN, 1.0 / (double)opts.embeddingDimension);
    geometricStretch *= opts.expansionStretch;
    for (int v = 0; v < newN; v++) {
        int parent = hierarchy->nodeLayers[currentLayer - 1][v].parentNode;
        ASSERT(parent < oldN, "Parent node " << parent << " is out of bounds " << oldN);
        // direct child count, not total contained leaves: children.size()^(1/d) spheres
        // tile the stretched layout (volume ~ newN) at unit density
        double numSiblings = hierarchy->nodeLayers[currentLayer][parent].children.size();

        tmpVec.setToRandomUnitVector();
        // clusters pack tighter than unit density since internal edges don't repel; measured
        // converged rms spreads: ~0.2*k^(1/d) at d=2, ~0.4 at d=4, ~0.5 at d=8, but end
        // quality and iteration count are insensitive to this factor in [0.15,1] at d=2 and 8
        constexpr double scatterPacking = 0.3;
        double sphere_size = scatterPacking * Toolkit::myPow(numSiblings, 1.0 / (double)opts.embeddingDimension);
        tmpVec *= sphere_size;
        for (int d = 0; d < opts.embeddingDimension; d++) {
            newPositions[v][d] = geometricStretch * oldPostions[parent][d] + tmpVec[d];
        }
    }

    currentLayer--;
    // initializeState=false skips the constructor's throwaway random coordinate/weight
    currentEmbedder = std::make_unique<WembedEmbedder>(hierarchy->graphs[currentLayer], opts, timer, false);
    currentEmbedder->setCoordinates(newPositions);
    currentEmbedder->setWeights(newWeights);

    timer->stopTiming("expanding");
}
