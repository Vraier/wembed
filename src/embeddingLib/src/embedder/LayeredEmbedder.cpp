#include "LayeredEmbedder.hpp"

#include "Macros.hpp"

void LayeredEmbedder::calculateStep() {
    currentIteration++;
    if (currentEmbedder.isFinished()) {
        expandPositions();
    }
    currentEmbedder.calculateStep();
}

bool LayeredEmbedder::isFinished() { return (currentLayer == 0) && currentEmbedder.isFinished(); }

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

void LayeredEmbedder::setCoordinates(const std::vector<std::vector<double>>& coordinates) {
    LOG_WARNING("Setting coordinates for layered embedder has no effect");
    unused(coordinates);
    return;
}

void LayeredEmbedder::setWeights(const std::vector<double>& weights) {
    LOG_WARNING("Setting weights for layered embedder has no effect");
    unused(weights);
    return;
}

std::vector<std::vector<double>> LayeredEmbedder::getCoordinates() { return currentEmbedder.getCoordinates(); }

std::vector<double> LayeredEmbedder::getWeights() { return currentEmbedder.getWeights(); }

std::vector<util::TimingResult> LayeredEmbedder::getTimings() { return timer->getHierarchicalTimingResults(); }

Graph LayeredEmbedder::getCurrentGraph() { return hierarchy->graphs[currentLayer]; }

void LayeredEmbedder::expandPositions() {
    LOG_INFO("Expanding positions to layer " << currentLayer - 1 << " in iteration " << currentIteration);
    timer->startTiming("expanding", "Expanding Positions");

    VecBuffer<1> buffer(opts.embeddingDimension);
    TmpVec<0> tmpVec(buffer);

    int newN = hierarchy->graphs[currentLayer - 1].getNumVertices();
    int oldN = hierarchy->graphs[currentLayer].getNumVertices();
    std::vector<std::vector<double>> oldPostions = currentEmbedder.getCoordinates();
    std::vector<std::vector<double>> newPositions(newN, std::vector<double>(opts.embeddingDimension, 0.0));
    ASSERT(oldN == oldPostions.size(), "Old positions size mismatch: " << oldN << " vs " << oldPostions.size());

    // calculate new weights
    std::vector<double> newWeights;
    if (opts.weightType == WeightType::Degree) {
        newWeights =
            NewWEmbedEmbedder::rescaleWeights(opts.dimensionHint, opts.embeddingDimension,
                                           NewWEmbedEmbedder::constructDegreeWeights(hierarchy->graphs[currentLayer - 1]));
    } else if (opts.weightType == WeightType::Unit) {
        newWeights = NewWEmbedEmbedder::constructUnitWeights(newN);
    } else {
        LOG_ERROR("Weight type not supported");
    }

    // calculate new positions
    double geometricStretch = Toolkit::myPow((double)newN / (double)oldN, 1.0 / (double)opts.embeddingDimension);
    geometricStretch *= opts.expansionStretch;
    for (int v = 0; v < newN; v++) {
        int parent = hierarchy->nodeLayers[currentLayer - 1][v].parentNode;
        ASSERT(parent < oldN, "Parent node " << parent << " is out of bounds " << oldN);
        double numSiblings = hierarchy->nodeLayers[currentLayer][parent].totalContainedNodes;

        tmpVec.setToRandomUnitVector();
        double sphere_size = Toolkit::myPow(numSiblings, 1.0 / (double)opts.embeddingDimension);
        tmpVec *= sphere_size; 
        for (int d = 0; d < opts.embeddingDimension; d++) {
            newPositions[v][d] = geometricStretch * oldPostions[parent][d] + tmpVec[d];
        }
    }

    currentLayer--;
    NewWEmbedEmbedder newEmbedder(hierarchy->graphs[currentLayer], opts, timer);
    currentEmbedder = std::move(newEmbedder);
    currentEmbedder.setCoordinates(newPositions);
    currentEmbedder.setWeights(newWeights);

    timer->stopTiming("expanding");
}
