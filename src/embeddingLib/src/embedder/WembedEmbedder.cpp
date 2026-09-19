#include "WembedEmbedder.hpp"

#include <algorithm>
#include <limits>

#include "LossFunction.hpp"
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
}

bool WembedEmbedder::isFinished() {
    if (this->state.currentIteration >= this->opts.maxIterations) return true;
    if (graphSize() <= 1) return true;
    if (this->opts.stopCriterion == StopCriterionType::Displacement) {
        return this->displacementMonitor->converged();
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
    if (this->opts.dynamicQueryBuffer != 0.0) {
        LOG_INFO("Dynamic queries: " << this->state.currentWeightedIndex.numRebuilds() << " rebuilds in "
                                     << this->state.currentWeightedIndex.numUpdates() << " index updates");
    }
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
    // discontinuous jump: cached repulsion candidates are stale
    state.lastMaxDisplacement = std::numeric_limits<double>::infinity();
}

void WembedEmbedder::setWeights(const std::vector<double> &weights) {
    ASSERT(graphSize() == weights.size());

    this->state.currentWeights = weights;
    sortNodes();

#pragma omp parallel for default(none) shared(invExpWeights, state) schedule(static)
    for (size_t i = 0; i < graphSize(); i++) {
        invExpWeights[i] = 1.0 / Toolkit::myPow(state.currentWeights[i], 1.0 / static_cast<double>(opts.embeddingDimension));
    }
    state.lastMaxDisplacement = std::numeric_limits<double>::infinity();
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

    const double weightV = state.currentWeights[v];
    const double weightU = state.currentWeights[u];

    //displace in random direction if positions are identical
    if (dist <= 0) {
        std::mt19937 gen = Rand::localGenerator(static_cast<uint32_t>(v), static_cast<uint32_t>(state.currentIteration));
        result.setToRandomUnitVector(gen);
        this->state.force[v] += result;
        // the repulsion pass counted this coincident pair; remove it from the objective
        if (WeightedIndex::ownsPair(weightV, weightU, v, u)) return -lossFunction::maxRepulsionLoss();
        return 0.0;
    }
    vectorOperations::differentiateLPNormDifference(posU, posV, dist, result);

    const double weightScaling = invExpWeights[v] * invExpWeights[u];
    const double weightedDist = dist * weightScaling;

    // repulsion skips the adjacency check and pushed near neighbors apart; pull with the
    // exact opposite (same weightedDist, same functions) and let the owner remove the
    // pair from the reported loss
    double forceFactor = lossFunction::attractionForceFactor(weightedDist) +
                         lossFunction::repulsionForceFactor(weightedDist);
    double lossContribution = lossFunction::attractionLoss(weightedDist);
    if (WeightedIndex::ownsPair(weightV, weightU, v, u)) {
        lossContribution -= lossFunction::repulsionLoss(weightedDist);
    }
    result *= forceFactor * weightScaling;

    this->state.force[v] += result;
    return lossContribution;
}

void WembedEmbedder::updateIndex() {
    state.currentWeightedIndex.update(this->state.currentPositions, this->state.currentWeights,
                                      this->state.lastMaxDisplacement);
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

double WembedEmbedder::pairRepulsion(const NodeId a, const NodeId b, TmpVec<0>& out) const {
    const CVecRef posA = state.currentPositions[a];
    const CVecRef posB = state.currentPositions[b];
    const double dist = vectorOperations::calculateLPNorm(posA, posB);

    // identical position get random kick
    if (dist <= 0) {
        std::mt19937 gen = Rand::localGenerator(static_cast<uint32_t>(a), static_cast<uint32_t>(state.currentIteration));
        out.setToRandomUnitVector(gen);
        return lossFunction::maxRepulsionLoss();
    }

    vectorOperations::differentiateLPNormDifference(posA, posB, dist, out);
    const double weightScaling = invExpWeights[a] * invExpWeights[b];
    const double weightedDist = dist * weightScaling;
    out *= lossFunction::repulsionForceFactor(weightedDist) * weightScaling;
    return lossFunction::repulsionLoss(weightedDist);
}

// Owner side can be computed directly.
// The owned side needs to compute the inverse mapping first.
// buckets are sorted by owner id, so every accumulation order is a function of the data and we get determinnism
void WembedEmbedder::calculateAllRepellingForces() {
    const size_t n = graphSize();
    if (!inCount) {
        inCount = std::make_unique<std::atomic<uint32_t>[]>(n);
        inCursor = std::make_unique<std::atomic<uint32_t>[]>(n);
        inOffset.resize(n + 1);
        ownedPairs.resize(n);
    }
    VecBuffer<1> buffer(this->opts.embeddingDimension);

    // phase 1: query, owner-side force + loss
    this->timer->startTiming("repel_p1", "P1 query + owner force");
#pragma omp parallel for default(none) firstprivate(buffer) shared(state, ownedPairs, lossPerNode) schedule(dynamic, 64)
    for (const NodeId v : state.sortedNodeIDs) {
        state.currentWeightedIndex.getOwnedRepellingPairs(v, ownedPairs[v]);
        double nodeLoss = 0.0;
        TmpVec<0> pairForce(buffer, 0.0);
        for (const NodeId u : ownedPairs[v]) {
            nodeLoss += pairRepulsion(v, u, pairForce);
            this->state.force[v] += pairForce;
        }
        this->lossPerNode[v] = nodeLoss;
    }
    this->timer->stopTiming("repel_p1");

    this->timer->startTiming("repel_transpose", "Transpose + sort");
    // inverse map: count, prefix, fill, then sort each bucket
#pragma omp parallel for default(none) firstprivate(n) schedule(static)
    for (size_t i = 0; i < n; i++) {
        inCount[i].store(0, std::memory_order_relaxed);
        inCursor[i].store(0, std::memory_order_relaxed);
    }
#pragma omp parallel for default(none) firstprivate(n) shared(ownedPairs) schedule(dynamic, 64)
    for (size_t v = 0; v < n; v++) {
        for (const NodeId u : ownedPairs[v]) {
            inCount[u].fetch_add(1, std::memory_order_relaxed);
        }
    }
    inOffset[0] = 0;
    for (size_t i = 0; i < n; i++) {
        inOffset[i + 1] = inOffset[i] + inCount[i].load(std::memory_order_relaxed);
    }
    inOwner.resize(inOffset[n]);
#pragma omp parallel for default(none) firstprivate(n) shared(ownedPairs, inOffset, inOwner) schedule(dynamic, 64)
    for (size_t v = 0; v < n; v++) {
        for (const NodeId u : ownedPairs[v]) {
            const uint64_t pos = inOffset[u] + inCursor[u].fetch_add(1, std::memory_order_relaxed);
            inOwner[pos] = static_cast<NodeId>(v);
        }
    }
#pragma omp parallel for default(none) firstprivate(n) shared(inOffset, inOwner) schedule(dynamic, 64)
    for (size_t u = 0; u < n; u++) {
        std::sort(inOwner.begin() + inOffset[u], inOwner.begin() + inOffset[u + 1]);
    }
    this->timer->stopTiming("repel_transpose");

    // phase 2: owned force, accumulated in ascending owner order
    this->timer->startTiming("repel_p2", "P2 target force");
#pragma omp parallel for default(none) firstprivate(buffer, n) shared(state, inOffset, inOwner) schedule(dynamic, 64)
    for (size_t u = 0; u < n; u++) {
        TmpVec<0> pairForce(buffer, 0.0);
        for (uint64_t i = inOffset[u]; i < inOffset[u + 1]; i++) {
            pairRepulsion(static_cast<NodeId>(u), inOwner[i], pairForce);
            this->state.force[u] += pairForce;
        }
    }
    this->timer->stopTiming("repel_p2");

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

    // max is order-independent, so this stays deterministic without a guarded reduction
    this->state.lastMaxDisplacement = *std::max_element(perNodeDisplacement.begin(), perNodeDisplacement.end());
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