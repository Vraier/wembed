#include "SimpleDotProductEmbedder.hpp"


// ======================================================================================
//
//                       PRIVATE FUNCTIONS SimpleDotProductEmbedder
//
// ======================================================================================


// ======================================================================================
//
//                       PUBLIC FUNCTIONS SimpleDotProductEmbedder
//
// ======================================================================================


void SimpleDotProductEmbedder::calculateStep() {
    //TODO:
}

bool SimpleDotProductEmbedder::isFinished() {
    if (this->state.currentIteration >= this->opts.maxIterations) return true;
    if (graphSize() <= 1) return true;
    switch (this->opts.stopCriterion) {
        case StopCriterionType::Displacement:
            return this->displacementMonitor->converged();
        case StopCriterionType::Loss:
            return this->convergenceMonitor->converged();
    }
    return this->convergenceMonitor->converged();
}

void SimpleDotProductEmbedder::calculateEmbedding() {
    LOG_INFO("Calculating embedding...");
    timer->startTiming("embedding_all", "Embedding");
    this->state.currentIteration = 0;
    while (!isFinished()) {
        calculateStep();
    }
    timer->stopTiming("embedding_all");
    LOG_INFO("Finished calculating embedding in iteration " << this->state.currentIteration);
}

Graph SimpleDotProductEmbedder::getCurrentGraph() {
    return this->graph;
}

std::vector<std::vector<double>> SimpleDotProductEmbedder::getCoordinates() {
    return this->state.currentPositions.convertToVector();
}

std::vector<double> SimpleDotProductEmbedder::getWeights() {
    return this->state.currentWeights;
}

std::vector<util::TimingResult> SimpleDotProductEmbedder::getTimings() {
    return timer->getHierarchicalTimingResults();
}

void SimpleDotProductEmbedder::setCoordinates(const std::vector<std::vector<double>> &coordinates) {
    const int coordDim = coordinates.empty() ? 0 : static_cast<int>(coordinates[0].size());
    ASSERT(graphSize() == coordinates.size());

    if (coordDim != this->opts.embeddingDimension)
        LOG_WARNING("Dimension of coordinates (" << coordDim << ") does not match embedding dimension ("
                                                 << opts.embeddingDimension << ")");

    for (size_t i = 0; i < graphSize(); i++) {
        ASSERT(coordinates[i].size() == coordDim,
               "coordinates[" << i << "].size()=" << coordinates[i].size() << ", dim=" << coordDim);
        for (int d = 0; d < std::min(this->opts.embeddingDimension, coordDim); d++) {
            state.currentPositions[i][d] = coordinates[i][d];
        }
    }
}

void SimpleDotProductEmbedder::setWeights(const std::vector<double> &weights) {
    ASSERT(graphSize() == weights.size());

    this->state.currentWeights = weights;
    sortNodes();

#pragma omp parallel for default(none) shared(invExpWeights, state) schedule(static)
    for (size_t i = 0; i < graphSize(); i++) {
        invExpWeights[i] = 1.0 / Toolkit::myPow(state.currentWeights[i], 1.0 / static_cast<double>(opts.embeddingDimension));
    }
}

std::vector<double> SimpleDotProductEmbedder::rescaleWeights(double dimensionHint, double embeddingDimension,
    const std::vector<double> &weights) {
    const auto N = static_cast<int>(weights.size());
    std::vector<double> rescaledWeights(N);

    for (NodeId v = 0; v < N; v++) {
        if (dimensionHint > 0) {
            rescaledWeights[v] = Toolkit::myPow(weights[v],
                                    static_cast<double>(embeddingDimension) / static_cast<double>(dimensionHint));
        } else {
            rescaledWeights[v] = weights[v];
        }
    }

    double weightSum = 0.0;
    for (int v = 0; v < N; v++) {
        weightSum += rescaledWeights[v];
    }
    for (int v = 0; v < N; v++) {
        rescaledWeights[v] = rescaledWeights[v] * (static_cast<double>(N) / weightSum);
    }
    return rescaledWeights;
}

std::vector<double> SimpleDotProductEmbedder::constructDegreeWeights(const Graph &g) {
    std::vector<double> weights(g.getNumVertices());
    for (NodeId v = 0; v < g.getNumVertices(); v++) {
        const int numNeighbors = g.getNumNeighbors(v);
        weights[v] = (numNeighbors > 0) ? numNeighbors : 1;
    }
    return weights;
}

std::vector<double> SimpleDotProductEmbedder::constructUnitWeights(int N) {
    std::vector<double> weights(N);
    for (NodeId v = 0; v < N; v++) {
        weights[v] = 1.0;
    }
    return weights;
}
