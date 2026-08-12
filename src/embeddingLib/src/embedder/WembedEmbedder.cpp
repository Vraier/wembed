#include "WembedEmbedder.hpp"

#include "ParallelReduce.hpp"
#include "VectorOperations.hpp"
#include "WeightedIndex.hpp"


// ======================================================================================
//
//                       PUBLIC FUNCTIONS WembedEmbedder
//
// ======================================================================================
void WembedEmbedder::calculateStep() {

    //Increase current step
    state.nextStep();

    //Abort in the case of the first hierarchy layer
    if (graphSize() <= 1) {
        return;
    }

    //Snapshot the positions so we can measure how far the nodes move this step.
    //state.currentPositions still holds the (recentred) positions from the previous step.
    this->previousPositions = this->state.currentPositions;

    //Rebuild indices
    this->timer->startTiming("index", "Construct spacial index");
    updateIndex();
    this->timer->stopTiming("index");

    //Compute attracting forces
    this->timer->startTiming("attracting_forces", "Compute Attracting Forces");
    calculateAllAttractingForces();
    this->timer->stopTiming("attracting_forces");

    //Compute repelling forces
    this->timer->startTiming("repelling_forces", "Compute Repelling Forces");
    calculateAllRepellingForces();
    this->timer->stopTiming("repelling_forces");

    //Compute centre forces
    if (this->opts.centreScale != 0.0) {
        this->timer->startTiming("centre_forces", "Computes Centre Force");
        calculateAllCentreForces();
        this->timer->stopTiming("centre_forces");
    }

    //Update positions
    this->timer->startTiming("apply_forces", "Applying Forces");
    const double learningRate = this->lrScheduler->learningRate(static_cast<int>(this->state.currentIteration));
    this->state.lastLearningRate = learningRate;
    this->posOptimizer->update(this->state.currentPositions, this->state.force, learningRate);
    this->timer->stopTiming("apply_forces");

    this->timer->startTiming("gravity", "Move graph towards centre");
    applyGravityCentre();
    this->timer->stopTiming("gravity");

    observeDisplacement();
    this->convergenceMonitor->observe(this->state.lastAttractLoss + this->state.lastRepelLoss);
    this->state.lastRelLossImprovement = this->convergenceMonitor->relImprovement();

    // if (this->state.currentIteration == 1 || (this->state.currentIteration > 0 && this->state.currentIteration % 10 == 0)) {
    //    std::cout << "(Iteration " << this->state.currentIteration << ": loss: "
    //              << this->state.lastAttractLoss + this->state.lastRepelLoss << ")" << std::endl;
    // S}
}

bool WembedEmbedder::isFinished() {
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

void WembedEmbedder::calculateEmbedding() {
    LOG_INFO("Calculating embedding...");
    timer->startTiming("embedding_all", "Embedding");
    this->state.currentIteration = 0;
    while (!isFinished()) {
        calculateStep();
    }
    timer->stopTiming("embedding_all");
    LOG_INFO("Finished calculating embedding in iteration " << this->state.currentIteration);
}

Graph WembedEmbedder::getCurrentGraph() {
     return this->graph;
}

std::vector<std::vector<double> > WembedEmbedder::getCoordinates() {
    return this->state.currentPositions.convertToVector();
}

std::vector<double> WembedEmbedder::getWeights() {
    return this->state.currentWeights;
}

std::vector<util::TimingResult> WembedEmbedder::getTimings() {
    return timer->getHierarchicalTimingResults();
}

void WembedEmbedder::setCoordinates(const std::vector<std::vector<double> > &coordinates) {
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

void WembedEmbedder::setWeights(const std::vector<double> &weights) {
    ASSERT(graphSize() == weights.size());

    this->state.currentWeights = weights;
    sortNodes();

#pragma omp parallel for default(none) shared(invExpWeights, state) schedule(static)
    for (size_t i = 0; i < graphSize(); i++) {
        invExpWeights[i] = 1.0 / Toolkit::myPow(state.currentWeights[i], 1.0 / static_cast<double>(opts.embeddingDimension));
    }
}

// ======================================================================================
//
//                       PRIVATE FUNCTIONS WembedEmbedder
//
// ======================================================================================


double WembedEmbedder::attractionForce(const NodeId v, const NodeId u, VecBuffer<1>& forceBuffer) {
    if (v == u) return 0.0;

    const CVecRef posV = state.currentPositions[v];
    const CVecRef posU = state.currentPositions[u];

    TmpVec<0> result(forceBuffer, 0.0);
    const double dist = vectorOperations::calculateLPNorm(posU, posV);

    //displace in random direction if positions are identical
    if (dist <= 0) {
        std::mt19937 gen = Rand::localGenerator(static_cast<uint32_t>(v), static_cast<uint32_t>(state.currentIteration));
        result.setToRandomUnitVector(gen);
        this->state.force[v] += result;
        return 0.0;
    }
    vectorOperations::differentiateLPNormDifference(posU, posV, dist, result);

    const double weightScaling = this->opts.additiveWeights ?
                           (invExpWeights[v] + invExpWeights[u]) :
                           (invExpWeights[v] * invExpWeights[u]);

    double lossContribution = 0.0;
    if (dist * weightScaling <= this->opts.edgeLength) {
        result *= 0;
    } else {
        result *= this->opts.attractionScale * weightScaling;
        lossContribution = dist - this->opts.edgeLength / weightScaling;
    }

    this->state.force[v] += result;
    return lossContribution;
}

double WembedEmbedder::repellingForce(const NodeId v, const NodeId u, VecBuffer<1>& forceBuffer) {
    if (v == u) return 0.0;

    const CVecRef posV = state.currentPositions[v];
    const CVecRef posU = state.currentPositions[u];
    TmpVec<0> result(forceBuffer, 0.0);
    const double dist = vectorOperations::calculateLPNorm(posV, posU);

    // displace in random direction if positions are identical (see attractionForce)
    if (dist <= 0) {
        std::mt19937 gen = Rand::localGenerator(static_cast<uint32_t>(v), static_cast<uint32_t>(state.currentIteration));
        result.setToRandomUnitVector(gen);
        this->state.force[v] += result;
        return 0.0;
    }

    vectorOperations::differentiateLPNormDifference(posV, posU, dist, result);

    // calculate weighted distance
    const double weightScaling = this->opts.additiveWeights ? (invExpWeights[v] + invExpWeights[u])
                                                            : (invExpWeights[v] * invExpWeights[u]);
    double lossContribution = 0.0;
    if (dist * weightScaling > this->opts.edgeLength) {
        result *= 0;
    } else {
        result *= this->opts.repulsionScale * weightScaling;
        lossContribution = this->opts.edgeLength / weightScaling - dist;
    }

    // increase repulsion force when we use less negative samples
    if (this->opts.numNegativeSamples > 0) {
        result *= static_cast<double>(graphSize()) / static_cast<double>(this->opts.numNegativeSamples);
    }

    this->state.force[v] += result;
    return lossContribution;
}

void WembedEmbedder::updateIndex() {
    if (this->opts.numNegativeSamples >= 0) {
        return; //we are not using a geometric index
    }

    //calculate new indices
    if (this->opts.IndexSize >= 1.0) {
        state.indexToGraphMap.resize(graphSize());
        std::iota(state.indexToGraphMap.begin(), state.indexToGraphMap.end(), 0);
        const std::vector<double> weightBuckets =
            WeightedIndex::getDoublingWeightBuckets(this->state.currentWeights, this->opts.doublingFactor);
        state.currentWeightedIndex.updateIndices(this->state.currentPositions, this->state.currentWeights, weightBuckets);
    } else {
        //Only insert a fraction of nodes into the index
        const int32_t numNodes = std::max(1, static_cast<int32_t>(graphSize() * this->opts.IndexSize));
        state.indexToGraphMap = Rand::randomSample(static_cast<int>(graphSize()), numNodes);
        VecList positions(this->opts.embeddingDimension, numNodes);
        std::vector<double> weights(numNodes);

#pragma omp parallel for default(none) shared(numNodes, positions, weights, state) schedule(static)
        for (size_t i = 0; i < numNodes; i++) {
            positions[i] = this->state.currentPositions[state.indexToGraphMap[i]];
            weights[i] = this->state.currentWeights[state.indexToGraphMap[i]];
        }

        const std::vector<double> weightBuckets = WeightedIndex::getDoublingWeightBuckets(weights, this->opts.doublingFactor);
        state.currentWeightedIndex.updateIndices(positions, weights, weightBuckets);
    }
}

std::vector<NodeId> WembedEmbedder::getRepellingCandidatesForNode(NodeId v, VecBuffer<2> &buffer) const {
    //TODO: Definitely think about refactoring this
    std::vector<NodeId> candidates;

    if (this->opts.numNegativeSamples >= 0) {
        candidates = sampleRandomNoise(std::min(static_cast<int32_t>(graphSize()), this->opts.numNegativeSamples));
        return candidates;
    }

    this->state.currentWeightedIndex.getNodesWithinWeightedDistance(this->state.currentPositions[v], this->state.currentWeights[v], this->opts.edgeLength,
                                                       candidates, buffer);
    for (NodeId& candidate: candidates) {
        candidate = this->state.indexToGraphMap[candidate];
        ASSERT(candidate < graphSize() && candidate >= 0, "Index out of bounds: " << candidate << " for N = " << graphSize());
    }
    return candidates;
}

void WembedEmbedder::calculateAllAttractingForces() {
    VecBuffer<1> buffer(this->opts.embeddingDimension);
#pragma omp parallel for default(none) firstprivate(buffer) shared(state, graph, lossPerNode) schedule(runtime)
    for (const NodeId v : this->state.sortedNodeIDs) {
        double nodeLoss = 0.0;
        for (const NodeId u : graph.getNeighbors(v)) {
            nodeLoss += attractionForce(v, u, buffer);
        }
        this->lossPerNode[v] = nodeLoss;
    }
    this->state.lastAttractLoss =
        util::deterministicSum(graphSize(), [this](std::size_t i) { return this->lossPerNode[i]; });
}

void WembedEmbedder::calculateAllRepellingForces() {
    VecBuffer<2> indexBuffer(this->opts.embeddingDimension);
    VecBuffer<1> forceBuffer(this->opts.embeddingDimension);
    numRepForceCalculations = 0;

#pragma omp parallel for default(none) firstprivate(indexBuffer, forceBuffer), shared(state, graph, lossPerNode), reduction(+:numRepForceCalculations), schedule(runtime)
    for (const NodeId v : state.sortedNodeIDs) {
        double nodeLoss = 0.0;
        const std::vector<NodeId> repellingCandidates = getRepellingCandidatesForNode(v, indexBuffer);
        for (const NodeId u : repellingCandidates) {
            if (graph.areNeighbors(v, u) || graph.areInSameColorClass(v, u)) {
                continue;
            }
            nodeLoss += repellingForce(v, u, forceBuffer);
            numRepForceCalculations++;
        }
        this->lossPerNode[v] = nodeLoss;
    }
    this->state.lastRepelLoss =
        util::deterministicSum(graphSize(), [this](std::size_t i) { return this->lossPerNode[i]; });
}

void WembedEmbedder::calculateAllCentreForces() {
#pragma omp parallel for default(none) shared(state, opts) schedule(static)
    for (const NodeId v : this->state.sortedNodeIDs) {
        this->state.force[v] += -1.0 * this->opts.centreScale * this->state.currentPositions[v];
    }
}

void WembedEmbedder::applyGravityCentre() {
    const int dim = this->opts.embeddingDimension;
    std::vector<double> dimGravity(dim, 0.0);
    for (int d = 0; d < dim; d++) {
        dimGravity[d] = util::deterministicSum(
                            graphSize(), [this, d](std::size_t v) { return this->state.currentPositions[static_cast<int>(v)][d]; }) /
                        static_cast<double>(graphSize());
    }
    // Wrap the centroid in a single-row VecList so we can use VecRef arithmetic below.
    // TODO: this can be a temp vec probably
    VecList gravityCentre({dimGravity});

#pragma omp parallel for default(none) shared(gravityCentre) schedule(static)
    for (size_t i = 0; i < graphSize(); i++) {
        this->state.currentPositions[i] -= gravityCentre[0];
    }
}

void WembedEmbedder::observeDisplacement() {
    const int dim = this->opts.embeddingDimension;
    const std::size_t n = graphSize();

    // Per-node movement since the snapshot and squared distance from the centre.
    // Each node is written by exactly one thread, then reduced deterministically,
    // so the resulting relative displacement is independent of the thread count.
#pragma omp parallel for default(none) firstprivate(dim, n) \
    shared(perNodeDisplacement, perNodeRadiusSq, state, previousPositions) schedule(static)
    for (std::size_t v = 0; v < n; v++) {
        const CVecRef pos = this->state.currentPositions[v];
        const CVecRef prev = this->previousPositions[v];
        this->perNodeDisplacement[v] = vectorOperations::calculateLPNorm(pos, prev);
        double radiusSq = 0.0;
        for (int d = 0; d < dim; d++) {
            radiusSq += pos[d] * pos[d];
        }
        this->perNodeRadiusSq[v] = radiusSq;
    }

    const double invN = 1.0 / static_cast<double>(n);
    const double meanDisplacement =
        util::deterministicSum(n, [this](std::size_t i) { return this->perNodeDisplacement[i]; }) * invN;
    const double meanRadiusSq =
        util::deterministicSum(n, [this](std::size_t i) { return this->perNodeRadiusSq[i]; }) * invN;
    const double radius = std::sqrt(meanRadiusSq);

    // guard a degenerate zero-radius layout (e.g. all nodes coincident)
    const double relDisplacement = radius > 0.0 ? meanDisplacement / radius : 0.0;
    this->state.lastRelDisplacement = relDisplacement;
    this->displacementMonitor->observe(relDisplacement);
}

//TODO: This could be moved somewhere else
std::vector<NodeId> WembedEmbedder::sampleRandomNoise(const int32_t numNodes) const {
    return Rand::randomSample(static_cast<int32_t>(graphSize()), numNodes);
}

std::vector<double> WembedEmbedder::rescaleWeights(const double dimensionHint, const double embeddingDimension,
                                                   const std::vector<double>& weights) {
    const int N = weights.size();
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

std::vector<double> WembedEmbedder::constructDegreeWeights(const Graph& g) {
    std::vector<double> weights(g.getNumVertices());
    for (NodeId v = 0; v < g.getNumVertices(); v++) {
        const int numNeighbors = g.getNumNeighbors(v);
        weights[v] = (numNeighbors > 0) ? numNeighbors : 1;
    }
    return weights;
}

std::vector<double> WembedEmbedder::constructUnitWeights(const int N) {
    std::vector<double> weights(N);
    for (NodeId v = 0; v < N; v++) {
        weights[v] = 1.0;
    }
    return weights;
}