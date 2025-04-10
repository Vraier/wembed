#include "WEmbedEmbedder.hpp"

void WEmbedEmbedder::calculateStep() {
    currentIteration++;

    if (N > 1'000'000 && currentIteration % 10 == 0) {
        std::cout << "(Iteration " << currentIteration << ")" << std::endl;
    }

    if (N <= 1) {
        // this happens in the first hierarchy layer
        insignificantPosChange = true;
        return;
    }

    currentForce.setAll(0);
    oldPositions.setAll(0);

    timer->startTiming("index", "Construct spacial index");
    updateIndex();  // sequential
    timer->stopTiming("index");

    // attracting forces
    timer->startTiming("attracting_forces", "Attracting Forces");
    calculateAllAttractingForces();  // parallel
    timer->stopTiming("attracting_forces");

    // repelling forces
    timer->startTiming("repelling_forces", "Repelling Forces");
    calculateAllRepellingForces();  // parallel
    timer->stopTiming("repelling_forces");

    // applying gradient
    timer->startTiming("apply_forces", "Applying Forces");
    // save old positions to calculate change later
#pragma omp parallel for schedule(static)
    for (int v = 0; v < N; v++) {
        oldPositions[v] = currentPositions[v];
    }
    // update positions based on force vector
    optimizer.update(currentPositions, currentForce);  // parallel
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

bool WEmbedEmbedder::isFinished() {
    bool isFinished = (currentIteration >= options.maxIterations) || insignificantPosChange;
    return isFinished;
}

void WEmbedEmbedder::calculateEmbedding() {
    LOG_INFO("Calculating embedding...");
    timer->startTiming("embedding_all", "Embedding");
    currentIteration = 0;
    optimizer.reset();
    while (!isFinished()) {
        calculateStep();
    }
    timer->stopTiming("embedding_all");
    LOG_INFO("Finished calculating embedding in iteration " << currentIteration);
}

Graph WEmbedEmbedder::getCurrentGraph() { return graph; }

std::vector<std::vector<double>> WEmbedEmbedder::getCoordinates() { return currentPositions.convertToVector(); }

std::vector<double> WEmbedEmbedder::getWeights() { return currentWeights; }

void WEmbedEmbedder::setCoordinates(const std::vector<std::vector<double>>& coordinates) {
    ASSERT(N == coordinates.size());
    currentPositions = VecList(coordinates);
}

void WEmbedEmbedder::setWeights(const std::vector<double>& weights) {
    ASSERT(N == weights.size());
    currentWeights = weights;

    // sort the node ids by weight
    sortedNodeIds.resize(N);
    std::iota(sortedNodeIds.begin(), sortedNodeIds.end(), 0);
    std::sort(sortedNodeIds.begin(), sortedNodeIds.end(),
              [this](int a, int b) { return currentWeights[a] > currentWeights[b]; });

    // calculate prefixSum
    weightPrefixSum.resize(N);
    weightPrefixSum[0] = currentWeights[0];
    for (int i = 1; i < N; i++) {
        weightPrefixSum[i] = weightPrefixSum[i - 1] + currentWeights[i];
    }
}

std::vector<util::TimingResult> WEmbedEmbedder::getTimings() { return timer->getHierarchicalTimingResults(); }

void WEmbedEmbedder::calculateAllAttractingForces() {
    VecBuffer<1> buffer(options.embeddingDimension);
#pragma omp parallel for firstprivate(buffer), schedule(runtime)
    for (NodeId v : sortedNodeIds) {
        for (NodeId u : graph.getNeighbors(v)) {
            attractionForce(v, u, buffer);
        }
    }
}

void WEmbedEmbedder::calculateAllRepellingForces() {
    // find nodes that are too close to each other
    VecBuffer<2> indexBuffer(options.embeddingDimension);
    VecBuffer<1> forceBuffer(options.embeddingDimension);

#pragma omp parallel for firstprivate(indexBuffer, forceBuffer), schedule(runtime)
    for (NodeId v : sortedNodeIds) {
        std::vector<NodeId> repellingCandidates = getRepellingCandidatesForNode(v, indexBuffer);
        for (NodeId u : repellingCandidates) {
            if (graph.areNeighbors(v, u) || graph.areInSameColorClass(v, u)) {
                continue;
            }
            repulstionForce(v, u, forceBuffer);
        }
    }
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

    if (weightDist <= options.edgeLength) {
        result *= 0;
    } else {
        result *= options.attractionScale / (Toolkit::myPow(wu * wv, 1.0 / options.embeddingDimension));
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

    if (weightDist > options.edgeLength) {
        result *= 0;
    } else {
        result *= options.repulsionScale / (Toolkit::myPow(wu * wv, 1.0 / options.embeddingDimension));
    }

    // increase repulsion force when we use less negative samples
    if (options.numNegativeSamples > 0) {
        result *= (double)N / (double)options.numNegativeSamples;
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

std::vector<double> WEmbedEmbedder::rescaleWeights(double dimensionHint, double embeddingDimension,
                                                   const std::vector<double>& weights) {
    const int N = weights.size();
    std::vector<double> rescaledWeights(N);

    for (NodeId v = 0; v < N; v++) {
        if (dimensionHint > 0) {
            rescaledWeights[v] = Toolkit::myPow(weights[v], (double)embeddingDimension / (double)dimensionHint);
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

void WEmbedEmbedder::updateIndex() {
    if (options.numNegativeSamples >= 0) {
        return;  // we are not using a geometric index
    }

    // calculate new indices
    if (options.IndexSize >= 1.0) {
        // insert all nodes into the index
        IndexToGraphMap.resize(N);
        std::iota(IndexToGraphMap.begin(), IndexToGraphMap.end(), 0);  // identity vector
        std::vector<double> weightBuckets =
            WeightedIndex::getDoublingWeightBuckets(currentWeights, options.doublingFactor);
        currentweightedIndex.updateIndices(currentPositions, currentWeights, weightBuckets);
    } else {
        // only insert a fraction of nodes into the index
        int numNodes = std::max(1, (int)(N * options.IndexSize));
        IndexToGraphMap = Rand::randomSample(N, numNodes);

        VecList positions(options.embeddingDimension, numNodes);
        std::vector<double> weights(numNodes);
        for (int i = 0; i < numNodes; i++) {
            positions[i] = currentPositions[IndexToGraphMap[i]];
            weights[i] = currentWeights[IndexToGraphMap[i]];
        }

        std::vector<double> weightBuckets = WeightedIndex::getDoublingWeightBuckets(weights, options.doublingFactor);
        currentweightedIndex.updateIndices(positions, weights, weightBuckets);
    }
}

std::vector<NodeId> WEmbedEmbedder::getRepellingCandidatesForNode(NodeId v, VecBuffer<2>& buffer) const {
    std::vector<NodeId> candidates;

    if (options.numNegativeSamples >= 0) {
        candidates = sampleRandomNodes(std::min(N, options.numNegativeSamples));
        return candidates;
    }

    if (options.useInfNorm) {
        currentweightedIndex.getNodesWithinWeightedInfNormDistance(currentPositions[v], currentWeights[v], options.edgeLength,
                                                           candidates, buffer);
    } else {
        currentweightedIndex.getNodesWithinWeightedDistance(currentPositions[v], currentWeights[v], options.edgeLength,
                                                    candidates, buffer);
    }

    // remap the candidates to the original graph indices
    for (NodeId& candidate : candidates) {
        candidate = IndexToGraphMap[candidate];
        ASSERT(candidate < N && candidate >= 0, "Index out of bounds: " << candidate << " for N = " << N);
    }
    return candidates;
}

std::vector<NodeId> WEmbedEmbedder::sampleRandomNodes(int numNodes) const {
    std::vector<NodeId> result;

    // sample node with probability proportional to the weight
    for (int i = 0; i < numNodes; i++) {
        double weightSample = Rand::randomDouble(0.0, weightPrefixSum.back());
        auto it = std::lower_bound(weightPrefixSum.begin(), weightPrefixSum.end(), weightSample);
        int index = std::distance(weightPrefixSum.begin(), it);
        ASSERT(index < N && index >= 0, "Index out of bounds: " << index << " for N = " << N);
        result.push_back(index);
    }
    return result;
}
