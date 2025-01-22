#include "WEmbedEmbedder.hpp"

void WEmbedEmbedder::calculateStep() {
    currentIteration++;
    const int N = graph.getNumVertices();

    currentForce.setAll(0);
    oldPositions.setAll(0);

    timer.startTiming("rTree", "Construct R-Tree");
    updateRTree();
    timer.stopTiming("rTree");

    // attracting forces
    timer.startTiming("attracting_forces", "Attracting Forces");
    calculateAllAttractingForces();
    timer.stopTiming("attracting_forces");

    // repelling forces
    timer.startTiming("repelling_forces", "Repelling Forces");
    calculateAllRepellingForces();
    timer.stopTiming("repelling_forces");

    // applying gradient
    timer.startTiming("apply_forces", "Applying Forces");
    // save old positions to calculate change later
    for (int v = 0; v < N; v++) {
        oldPositions[v] = currentPositions[v];
    }
    // update positions based on force vector
    optimizer.update(currentPositions, currentForce);
    timer.stopTiming("apply_forces");

    // calculate change in position
    timer.startTiming("position_change", "Change in Positions");
    VecBuffer<1> buffer(options.embeddingDimension);
    TmpVec<0> tmpVec(buffer);
    double sumNormSquared = 0;
    double sumNormDiffSquared = 0;
    for (int v = 0; v < N; v++) {
        sumNormSquared += oldPositions[v].sqNorm();
        tmpVec = oldPositions[v] - currentPositions[v];
        sumNormDiffSquared += tmpVec.sqNorm();
    }
    if ((sumNormDiffSquared / sumNormSquared) < options.relativePosMinChange) {
        insignificantPosChange = true;
    }
    timer.stopTiming("position_change");
}

bool WEmbedEmbedder::isFinished() {
    bool isFinished = (currentIteration >= options.maxIterations) || insignificantPosChange;
    return isFinished;
}

void WEmbedEmbedder::calculateEmbedding() {
    LOG_INFO("Calculating embedding...");
    timer.startTiming("embedding_all", "Embedding");
    currentIteration = 0;
    optimizer.reset();
    while (!isFinished()) {
        calculateStep();
    }
    timer.stopTiming("embedding_all");
    LOG_INFO("Finished calculating embedding in iteration " << currentIteration);
}

Graph WEmbedEmbedder::getCurrentGraph() { return graph; }

std::vector<std::vector<double>> WEmbedEmbedder::getCoordinates() { return currentPositions.convertToVector(); }

std::vector<double> WEmbedEmbedder::getWeights() { return currentWeights; }

void WEmbedEmbedder::setCoordinates(const std::vector<std::vector<double>>& coordinates) {
    ASSERT(graph.getNumVertices() == coordinates.size());
    currentPositions = VecList(coordinates);
}

void WEmbedEmbedder::setWeights(const std::vector<double>& weights) {
    ASSERT(graph.getNumVertices() == weights.size());
    currentWeights = weights;
}

std::vector<util::TimingResult> WEmbedEmbedder::getTimings() { return timer.getHierarchicalTimingResults(); }

void WEmbedEmbedder::calculateAllAttractingForces() {
    VecBuffer<1> buffer(options.embeddingDimension);
#pragma omp parallel for firstprivate(buffer)
    for (NodeId v = 0; v < graph.getNumVertices(); v++) {
        for (NodeId u : graph.getNeighbors(v)) {
            attractionForce(v, u, buffer);
        }
    }
}

void WEmbedEmbedder::calculateAllRepellingForces() {
    // find nodes that are too close to each other
    timer.startTiming("candidates", "Finding Repelling Candidates");
    // TODO: nodes with larger weight should be treated first
    std::vector<std::vector<NodeId>> repellingCandidates(graph.getNumVertices());
    VecBuffer<2> rTreeBuffer(options.embeddingDimension);
#pragma omp parallel for firstprivate(rTreeBuffer)
    for (NodeId v = 0; v < graph.getNumVertices(); v++) {
        repellingCandidates[v] = getRepellingCandidatesForNode(v, rTreeBuffer);
    }
    timer.stopTiming("candidates");

    timer.startTiming("sum_of_forces", "Summing repelling Forces for each Node");
    VecBuffer<1> forceBuffer(options.embeddingDimension);
#pragma omp parallel for firstprivate(forceBuffer)
    for (NodeId v = 0; v < graph.getNumVertices(); v++) {
        for (NodeId u : repellingCandidates[v]) {
            if (options.neighborRepulsion || !graph.areNeighbors(v, u)) {
                repulstionForce(v, u, forceBuffer);
            }
        }
    }
    timer.stopTiming("sum_of_forces");
}

void WEmbedEmbedder::attractionForce(int v, int u, VecBuffer<1>& buffer) {
    if (v == u) return;

    CVecRef posV = currentPositions[v];
    CVecRef posU = currentPositions[u];
    TmpVec<0> result(buffer, 0.0);
    result = posU - posV;

    double dist;
    if (options.useInfNorm) {
        dist = result.infNorm();
    } else {
        dist = result.norm();
    }
    // displace in random direction if positions are identical
    if (dist <= 0) {
        result.setToRandomUnitVector();
        currentForce[v] += result;
        return;
    }

    if (options.useInfNorm) {
        result.infNormed();
    } else {
        result.normed();
    }

    // calculate weighted distance
    double wv = currentWeights[v];
    double wu = currentWeights[u];
    double weightDist = dist / Toolkit::myPow(wu * wv, 1.0 / options.embeddingDimension);

    if (weightDist <= options.sigmoidLength) {
        result *= 0;
    } else {
        result *= options.sigmoidScale / (Toolkit::myPow(wu * wv, 1.0 / options.embeddingDimension));
    }

    currentForce[v] += result;
}

void WEmbedEmbedder::repulstionForce(int v, int u, VecBuffer<1>& buffer) {
    if (v == u) return;

    CVecRef posV = currentPositions[v];
    CVecRef posU = currentPositions[u];
    TmpVec<0> result(buffer, 0.0);
    result = posV - posU;

    double dist;
    if (options.useInfNorm) {
        dist = result.infNorm();
    } else {
        dist = result.norm();
    }

    // displace in random direction if positions are identical
    if (dist <= 0) {
        result.setToRandomUnitVector();
        currentForce[v] += result;
        return;
    }

    if (options.useInfNorm) {
        result.infNormed();
    } else {
        result.normed();
    }

    // calculate weighted distance
    double wv = currentWeights[v];
    double wu = currentWeights[u];
    double weightDist = dist / Toolkit::myPow(wu * wv, 1.0 / options.embeddingDimension);

    if (weightDist > options.sigmoidLength) {
        result *= 0;
    } else {
        result *= options.sigmoidScale / (Toolkit::myPow(wu * wv, 1.0 / options.embeddingDimension));
    }

    currentForce[v] += result;
}

std::vector<double> WEmbedEmbedder::constructDegreeWeights(const Graph& g) {
    std::vector<double> weights(g.getNumVertices());
    for (NodeId v = 0; v < g.getNumVertices(); v++) {
        weights[v] = g.getNumNeighbors(v);
    }
    return weights;
}

std::vector<double> WEmbedEmbedder::constructUnitWeights(int N) {
    std::vector<double> weights(N);
    for (NodeId v = 0; v < N; v++) {
        weights[v] = 1.0;
    }
    return weights;
}

std::vector<double> WEmbedEmbedder::rescaleWeights(int dimensionHint, int embeddingDimension,
                                                   const std::vector<double>& weights) {
    const int N = weights.size();
    std::vector<double> rescaledWeights(N);

    for (NodeId v = 0; v < N; v++) {
        if (dimensionHint > 0) {
            rescaledWeights[v] = Toolkit::myPow(weights[v], (double)embeddingDimension/ (double)dimensionHint);
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

std::vector<std::vector<double>> WEmbedEmbedder::constructRandomCoordinates(int dimension, int N) {
    const double CUBE_SIDE_LENGTH = Toolkit::myPow(N, 1.0 / dimension);
    std::vector<std::vector<double>> coords(N, std::vector<double>(dimension));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < dimension; j++) {
            coords[i][j] = Rand::randomDouble(0, CUBE_SIDE_LENGTH);
        }
    }
    return coords;
}

void WEmbedEmbedder::updateRTree() {
    currentRTree = WeightedRTree(options.embeddingDimension);
    std::vector<double> weightBuckets = WeightedRTree::getDoublingWeightBuckets(currentWeights, options.doublingFactor);
    currentRTree.updateRTree(currentPositions, currentWeights, weightBuckets);
}

std::vector<NodeId> WEmbedEmbedder::getRepellingCandidatesForNode(NodeId v, VecBuffer<2>& buffer) const {
    std::vector<NodeId> candidates;
    for (size_t w_class = 0; w_class < currentRTree.getNumWeightClasses(); w_class++) {
        std::vector<NodeId> tmp;
        if (options.useInfNorm) {
            currentRTree.getNodesWithinWeightedDistanceInfNormForClass(currentPositions[v], currentWeights[v],
                                                                       options.sigmoidLength, w_class, tmp, buffer);
        }
        currentRTree.getNodesWithinWeightedDistanceForClass(currentPositions[v], currentWeights[v],
                                                            options.sigmoidLength, w_class, tmp, buffer);
        for (NodeId u : tmp) {
            if (v != u) {
                candidates.push_back(u);
            }
        }
    }
    return candidates;
}
