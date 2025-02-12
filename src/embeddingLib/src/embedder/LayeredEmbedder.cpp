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

    VecBuffer<1> buffer(options.embeddingDimension);
    TmpVec<0> tmpVec(buffer);

    int newN = hierarchy->graphs[currentLayer - 1].getNumVertices();
    int oldN = hierarchy->graphs[currentLayer].getNumVertices();
    std::vector<std::vector<double>> oldPostions = currentEmbedder.getCoordinates();
    std::vector<std::vector<double>> newPositions(newN);

    // calculate new weights
    std::vector<double> newWeights;
    if (options.weightType == WeightType::Degree) {
        newWeights =
            WEmbedEmbedder::rescaleWeights(options.dimensionHint, options.embeddingDimension,
                                           WEmbedEmbedder::constructDegreeWeights(hierarchy->graphs[currentLayer - 1]));
    } else if (options.weightType == WeightType::Unit) {
        newWeights = WEmbedEmbedder::constructUnitWeights(newN);
    } else {
        LOG_ERROR("Weight type not supported");
    }

    // calculate new positions
    double stretch = Toolkit::myPow((double)newN / (double)oldN, 1.0 / (double)options.embeddingDimension);
    for (int v = 0; v < newN; v++) {
        int parent = hierarchy->nodeLayers[currentLayer - 1][v].parentNode;
        ASSERT(parent < oldPostions.size(), "Parent node " << parent << " is out of bounds " << oldPostions.size());

        tmpVec.setToRandomUnitVector();
        tmpVec *= 0.1;
        newPositions[v] = oldPostions[parent];
        for (int d = 0; d < options.embeddingDimension; d++) {
            newPositions[v][d] += tmpVec[d];
            newPositions[v][d] *= stretch;
        }
    }

    currentLayer--;
    SingleLayerEmbedder newEmbedder(hierarchy, currentLayer, options, timer);
    currentEmbedder = std::move(newEmbedder);
    currentEmbedder.setCoordinates(newPositions);
    currentEmbedder.setWeights(newWeights);

    timer->stopTiming("expanding");
}

void SingleLayerEmbedder::calculateStep() {
    currentIteration++;
    const int N = hierarchy->getLayerSize(LAYER);

    if (N > 1000000 && currentIteration % 10 == 0) {
        std::cout << "(Iteration " << currentIteration << ")" << std::endl;
    }

    if (N <= 1) {
        // this happens in the first hierarchy layer
        insignificantPosChange = true;
        return;
    }

    currentForce.setAll(0);
    oldPositions.setAll(0);

    // rebuid the rTree with new positions
    timer->startTiming("rTree", "Construct R-Tree");
    updateRTree();  // sequential
    timer->stopTiming("rTree");

    // calculate forces
    timer->startTiming("attracting_forces", "Attracting Forces");
    calculateAllAttractingForces();  // parallel
    timer->stopTiming("attracting_forces");

    timer->startTiming("repelling_forces", "Repelling Forces");
    calculateAllRepellingForces();  // parallel
    timer->stopTiming("repelling_forces");

    // save old positions to calculate change later
    timer->startTiming("apply_forces", "Applying Forces");
#pragma omp parallel for schedule(static)
    for (int v = 0; v < N; v++) {
        oldPositions[v] = currentPositions[v];
    }

    // update positions based on force vector
    optimizer.update(currentPositions, currentForce);
    timer->stopTiming("apply_forces");

    // calculate change in position
    timer->startTiming("position_change", "Change in Positions");
    VecBuffer<1> buffer(options.embeddingDimension);
    double sumNormSquared = 0;
    double sumNormDiffSquared = 0;

#pragma omp parallel for reduction(+ : sumNormSquared, sumNormDiffSquared), firstprivate(buffer), schedule(static)
    for (int v = 0; v < N; v++) {
        TmpVec<0> tmpVec(buffer);
        tmpVec = oldPositions[v] - currentPositions[v];
        sumNormSquared += oldPositions[v].sqNorm();
        sumNormDiffSquared += tmpVec.sqNorm();
    }

    if ((sumNormDiffSquared / sumNormSquared) < options.relativePosMinChange) {
        insignificantPosChange = true;
    }
    timer->stopTiming("position_change");
}

bool SingleLayerEmbedder::isFinished() {
    bool isFinished = (currentIteration >= options.maxIterations) || insignificantPosChange;

    return isFinished;
}

void SingleLayerEmbedder::calculateEmbedding() {
    currentIteration = 0;
    optimizer.reset();
    while (!isFinished()) {
        calculateStep();
    }
}

std::vector<std::vector<double>> SingleLayerEmbedder::getCoordinates() { return currentPositions.convertToVector(); }

std::vector<double> SingleLayerEmbedder::getWeights() { return currentWeights; }

void SingleLayerEmbedder::setCoordinates(const std::vector<std::vector<double>>& coordinates) {
    ASSERT(N == coordinates.size());
    currentPositions = VecList(coordinates);
}

void SingleLayerEmbedder::setWeights(const std::vector<double>& weights) {
    ASSERT(N == weights.size());
    currentWeights = weights;

    // sort the node ids by weight
    sortedNodeIds.resize(N);
    std::iota(sortedNodeIds.begin(), sortedNodeIds.end(), 0);
    std::sort(sortedNodeIds.begin(), sortedNodeIds.end(),
              [this](int a, int b) { return currentWeights[a] > currentWeights[b]; });
}

void SingleLayerEmbedder::calculateAllAttractingForces() {
    VecBuffer<1> buffer(options.embeddingDimension);
#pragma omp parallel for firstprivate(buffer), schedule(runtime)
    for (NodeId v : sortedNodeIds) {
        for (NodeId u : graph.getNeighbors(v)) {
            attractionForce(v, u, buffer);
        }
    }
}

void SingleLayerEmbedder::calculateAllRepellingForces() {
    // find nodes that are too close to each other
    VecBuffer<2> rTreeBuffer(options.embeddingDimension);
    VecBuffer<1> forceBuffer(options.embeddingDimension);
#pragma omp parallel for firstprivate(rTreeBuffer, forceBuffer), schedule(runtime)
    for (NodeId v : sortedNodeIds) {
        std::vector<NodeId> repellingCandidates = getRepellingCandidatesForNode(v, rTreeBuffer);
        for (NodeId u : repellingCandidates) {
            if (graph.areNeighbors(v, u) || graph.areInSameColorClass(v, u)) {
                continue;
            }
            repulstionForce(v, u, forceBuffer);
        }
    }
}

void SingleLayerEmbedder::attractionForce(int v, int u, VecBuffer<1>& buffer) {
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

void SingleLayerEmbedder::repulstionForce(int v, int u, VecBuffer<1>& buffer) {
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

void SingleLayerEmbedder::updateRTree() {
    currentRTree = WeightedRTree(options.embeddingDimension);
    std::vector<double> weightBuckets = WeightedRTree::getDoublingWeightBuckets(currentWeights, options.doublingFactor);
    currentRTree.updateRTree(currentPositions, currentWeights, weightBuckets);
}

std::vector<NodeId> SingleLayerEmbedder::getRepellingCandidatesForNode(NodeId v, VecBuffer<2>& buffer) const {
    std::vector<NodeId> candidates;
    for (size_t w_class = 0; w_class < currentRTree.getNumWeightClasses(); w_class++) {
        if (options.useInfNorm) {
            currentRTree.getNodesWithinWeightedDistanceInfNormForClass(
                currentPositions[v], currentWeights[v], options.sigmoidLength, w_class, candidates, buffer);
        }
        currentRTree.getNodesWithinWeightedDistanceForClass(currentPositions[v], currentWeights[v],
                                                            options.sigmoidLength, w_class, candidates, buffer);
    }
    return candidates;
}
