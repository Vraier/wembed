#include "AbstractSimpleEmbedder.hpp"

#include <omp.h>

#include <algorithm>

#include "AdamOptimizer.hpp"
#include "Graph.hpp"
#include "SimpleOptimizer.hpp"

Graph AbstractSimpleEmbedder::getCurrentGraph() { return graph.getGraph(); }

std::vector<std::vector<double>> AbstractSimpleEmbedder::getCoordinates() {
    return graph.coordinates.convertToVector();
}

std::vector<double> AbstractSimpleEmbedder::getWeights() { return graph.getAllNodeWeights(); }

void AbstractSimpleEmbedder::setCoordinates(const std::vector<std::vector<double>>& coordinates) {
    ASSERT(graph.getNumVertices() == coordinates.size());
    graph.coordinates = VecList(coordinates);
}

void AbstractSimpleEmbedder::setWeights(const std::vector<double>& weights) {
    ASSERT(graph.getNumVertices() == weights.size());
    for (NodeId v = 0; v < graph.getNumVertices(); v++) {
        graph.setNodeWeight(v, weights[v]);
    }
}

void AbstractSimpleEmbedder::initializeOptimizer() {
    const int N = graph.getNumVertices();
    if (options.optimizerType == Simple) {
        optimizer = std::make_unique<SimpleOptimizer>(options.embeddingDimension, N, options.speed,
                                                      options.coolingFactor, options.maxDisplacement);
    } else if (options.optimizerType == Adam) {
        optimizer = std::make_unique<AdamOptimizer>(options.embeddingDimension, N, options.speed, options.coolingFactor,
                                                    0.9, 0.999, 10e-8);
    } else {
        LOG_ERROR("Unknown optimizer type");
    }
    optimizer->reset();
}

void AbstractSimpleEmbedder::calculateEmbedding() {
    LOG_INFO("Calculating layout...");
    currIteration = 0;
    numForceSteps = 0;
    numWeightSteps = 0;

    // Run embedding with simulation
    // if (options.animate) {
    //    SFMLDrawer animation;
    //    while (!isFinished()) {
    //        calculateStep();
    //
    //        timer.startTiming("animate", "Animation");
    //        animation.processFrame(graph.getGraph(),
    //        Common::projectOntoPlane(getCurrentLayout//().convertToVector())); timer.stopTiming("animate");
    //    }
    //}
    // run embedding without simulation
    // else {
    while (!isFinished()) {
        calculateStep();
    }
    //}

    LOG_INFO("Finished calculating layout in iteration " << currIteration << " after " << numForceSteps
                                                         << " force steps and " << numWeightSteps << " weight steps");
    // dumpDebugAtTermination();
}

void AbstractSimpleEmbedder::calculateStep() {
    if (insignificantPosChange && !options.staticWeights) {
        insignificantPosChange = false;
        calculateWeightStep();
    } else {
        calculateForceStep();
    }

    currIteration++;
}

bool AbstractSimpleEmbedder::isFinished() {
    bool isFinished = (currIteration >= options.maxIterations) || insignificantWeightChange ||
                      (options.staticWeights && insignificantPosChange);
    return isFinished;
}

void AbstractSimpleEmbedder::dumpDebugAtTermination() { return; }

std::vector<util::TimingResult> AbstractSimpleEmbedder::getTimings() const {
    return timer.getHierarchicalTimingResults();
}

std::vector<double> AbstractSimpleEmbedder::constructDegreeWeights(const Graph& g) {
    std::vector<double> weights(g.getNumVertices());
    for (NodeId v = 0; v < g.getNumVertices(); v++) {
        weights[v] = g.getNumNeighbors(v);
    }
    return weights;
}

std::vector<double> AbstractSimpleEmbedder::constructUnitWeights(int N) {
    std::vector<double> weights(N);
    for (NodeId v = 0; v < N; v++) {
        weights[v] = 1.0;
    }
    return weights;
}

std::vector<double> AbstractSimpleEmbedder::rescaleWeights(int dimensionHint, int embeddingDimension,
                                                           const std::vector<double>& weights) {
    const int N = weights.size();
    std::vector<double> rescaledWeights(N);

    for (NodeId v = 0; v < N; v++) {
        if (dimensionHint > 0) {
            rescaledWeights[v] = std::pow(weights[v], (double)embeddingDimension / (double)dimensionHint);
        } else {
            rescaledWeights[v] = weights[v];
        }
    }

    double weightSum = 0.0;
    for (int v = 0; v < N; v++) {
        weightSum += rescaledWeights[v];
    }
    for (int v = 0; v < N; v++) {
        rescaledWeights[v] = rescaledWeights[v] * ((double)N / weightSum);
    }
    return rescaledWeights;
}

std::vector<std::vector<double>> AbstractSimpleEmbedder::constructRandomCoordinates(int dimension, int N) {
    const double CUBE_SIDE_LENGTH = std::pow(N, 1.0 / dimension);
    std::vector<std::vector<double>> coords(N, std::vector<double>(dimension));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < dimension; j++) {
            coords[i][j] = Rand::randomDouble(0, CUBE_SIDE_LENGTH);
        }
    }
    return coords;
}

void AbstractSimpleEmbedder::calculateForceStep() {
    numForceSteps++;
    TmpVec<FORCE_STEP_BUFFER> tmpVec(buffer, 0.0);
    const int N = graph.getNumVertices();

    currentForce.setAll(0);
    oldPositions.setAll(0);

    // calculate new forces
    timer.startTiming("attracting_forces", "Calculate Attracting Forces");
    calculateAllAttractingForces();
    timer.stopTiming("attracting_forces");

    timer.startTiming("repelling_forces", "Calculate Repelling Forces");
    calculateAllRepellingForces();
    timer.stopTiming("repelling_forces");

    timer.startTiming("apply_forces", "Apply Forces");

    for (int v = 0; v < N; v++) {
        oldPositions[v] = graph.coordinates[v];
    }

    optimizer->update(graph.coordinates, currentForce);

    // calculate change in position
    double sumNormSquared = 0;
    double sumNormDiffSquared = 0;
    for (int v = 0; v < N; v++) {
        sumNormSquared += oldPositions[v].sqNorm();
        tmpVec = oldPositions[v] - graph.coordinates[v];
        sumNormDiffSquared += tmpVec.sqNorm();
    }
    if ((sumNormDiffSquared / sumNormSquared) < options.relativePosMinChange) {
        insignificantPosChange = true;
    }
    timer.stopTiming("apply_forces");
}

void AbstractSimpleEmbedder::calculateAllAttractingForces() {
    for (NodeId v = 0; v < graph.getNumVertices(); v++) {
        for (NodeId u : graph.getNeighbors(v)) {
            currentForce[v] += attractionForce(v, u);
        }
    }
}

void AbstractSimpleEmbedder::calculateAllRepellingForces() {
    for (NodeId v = 0; v < graph.getNumVertices(); v++) {
        for (NodeId u = 0; u < graph.getNumVertices(); u++) {
            if (options.neighborRepulsion || !graph.areNeighbors(v, u)) {
                currentForce[v] += repulsionForce(v, u);
            }
        }
    }
}

void AbstractSimpleEmbedder::calculateWeightStep() {
    numWeightSteps++;
    const int N = graph.getNumVertices();

    double currCooling = std::pow(options.weightCooling, numWeightSteps);
    for (NodeId v = 0; v < N; v++) {
        currentWeightForce[v] = (sumWeightRepulsionForce(v) + sumWeightAttractionForce(v));
        currentWeightForce[v] *= options.weightSpeed * currCooling;

        // cap the maximum replacement of the node
        currentWeightForce[v] = std::max(currentWeightForce[v], -options.maxDisplacement);
        currentWeightForce[v] = std::min(currentWeightForce[v], options.maxDisplacement);
    }

    double sumTmpWeight = 0.0;
    // apply movement based on force
    for (NodeId v = 0; v < N; v++) {
        oldWeights[v] = graph.getNodeWeight(v);
        newWeights[v] = oldWeights[v] + currentWeightForce[v];

        sumTmpWeight += newWeights[v];
    }

    // normalize weights and update them in graph
    for (NodeId a = 0; a < N; a++) {
        newWeights[a] *= ((double)N / sumTmpWeight);
        // NOTE(JP) scaling the weights seem to perform better
        graph.setNodeWeight(a, newWeights[a]);
    }

    double sumWeightDiff = 0.0;
    double sumNewWeights = 0.0;
    for (NodeId a = 0; a < N; a++) {
        sumWeightDiff += std::abs(newWeights[a] - oldWeights[a]);
        sumNewWeights += newWeights[a];
    }
    double relativeWeightChange = sumWeightDiff / sumNewWeights;

    if (relativeWeightChange < options.relativeWeightMinChange) {
        // LOG_DEBUG( "Insignificant weight change of " << sumWeightDiff / sumNewWeights);
        insignificantWeightChange = true;
    }
}

double AbstractSimpleEmbedder::sumWeightRepulsionForce(NodeId v) {
    double sumRepForce = 0.0;
    for (NodeId u = 0; u < graph.getNumVertices(); u++) {
        sumRepForce += weightRepulsionForce(v, u);
    }
    return sumRepForce;
}

double AbstractSimpleEmbedder::sumWeightAttractionForce(NodeId v) {
    double sumAttrForce = 0.0;
    for (NodeId u : graph.getNeighbors(v)) {
        sumAttrForce += weightAttractionForce(v, u);
    }
    return sumAttrForce;
}

double AbstractSimpleEmbedder::weightRepulsionForce(NodeId v, NodeId u) {
    unused(v);
    unused(u);
    return 0.0;
}

double AbstractSimpleEmbedder::weightAttractionForce(NodeId v, NodeId u) {
    unused(v);
    unused(u);
    return 0.0;
}

VecList AbstractSimpleEmbedder::getCurrentLayout() { return graph.coordinates; }

EmbeddedGraph AbstractSimpleEmbedder::getEmbeddedGraph() { return graph; }
