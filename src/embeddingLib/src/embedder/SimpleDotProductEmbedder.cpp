#include "SimpleDotProductEmbedder.hpp"

#include "ParallelReduce.hpp"
#include "VectorOperations.hpp"


// ======================================================================================
//
//                       PRIVATE FUNCTIONS SimpleDotProductEmbedder
//
// ======================================================================================

void SimpleDotProductEmbedder::calculateAllAttractingForces() {
    VecBuffer<1> buffer(this->opts.embeddingDimension);
#pragma omp parallel for default(none) firstprivate(buffer) shared(state, graph, lossPerNode) schedule(runtime)
    for (const NodeId v : this->state.sortedNodeIDs) {
        double nodeLoss = 0.0;
        for (const NodeId u : graph.getNeighbors(v)) {
            nodeLoss += attractionForce(v, u, buffer);
        }
        this->lossPerNode[v] = nodeLoss;
    }
    const double loss = util::deterministicSum(graphSize(), [this](std::size_t i) { return this->lossPerNode[i]; });
    this->state.lastAttractLoss = loss;
}

void SimpleDotProductEmbedder::calculateAllRepellingForces() {
    VecBuffer<1> buffer(this->opts.embeddingDimension);
    numRepForceCalculations = 0;

#pragma omp parallel for default(none) firstprivate(buffer) shared(state, graph, lossPerNode) reduction(+:numRepForceCalculations) schedule(runtime)
    for (const NodeId v : this->state.sortedNodeIDs) {
        double nodeLoss = 0.0;
        for (const NodeId u : this->state.sortedNodeIDs) {
            if (v == u || this->graph.areNeighbors(v, u)) {
                continue;
            }
            nodeLoss += repellingForce(v, u, buffer);
            numRepForceCalculations++;
        }
        this->lossPerNode[v] = nodeLoss;
    }

    const double loss = util::deterministicSum(graphSize(), [this](std::size_t i){return this->lossPerNode[i];});
    this->state.lastRepelLoss = loss;
}

void SimpleDotProductEmbedder::calculateAllCentreForces() {
#pragma omp parallel for default(none) shared(state, opts) schedule(static)
    for (const NodeId v : this->state.sortedNodeIDs) {
        this->state.force[v] += -1.0 * this->opts.centreScale * this->state.currentPositions[v];
    }
}

double SimpleDotProductEmbedder::attractionForce(NodeId v, NodeId u, VecBuffer<1> &forceBuffer) {
    if (v == u) return 0.0;

    const CVecRef posV = state.currentPositions[v];
    const CVecRef posU = state.currentPositions[u];

    TmpVec<0> result(forceBuffer, 0.0);
    const double dist = vectorOperations::calculateDotProductNorm(posU, posV);

    //displace in random direction if positions are identical
    VecBuffer<2> norms(posV.dimension());
    TmpVec<0> tmpV(norms);
    TmpVec<1> tmpU(norms);
    tmpV = posV.norm() * posV;
    tmpU = posU.norm() * posU;
    bool isSameDirection = true;
    for (int i = 0; i < tmpV.dimension(); i++) {
        if (tmpV[i] != tmpU[i]) {
            isSameDirection = false;
            break;
        }
    }

    if (isSameDirection) {
        std::mt19937 gen = Rand::localGenerator(static_cast<uint32_t>(v), static_cast<uint32_t>(state.currentIteration));
        VecBuffer<1> displace(posU.dimension());
        TmpVec<0> tmpDis(displace);
        tmpDis.setToRandomUnitVector(gen);
        result = posV + 0.05 * tmpDis;
        this->state.force[v] += result;
        return 0.0;
    }

    vectorOperations::differentiateDotProductNorm(posU, posV, dist, result);

    const double lossContribution = dist - this->opts.edgeLength;
    if (dist >= this->opts.edgeLength) { // if dot product distance it greater or equal to 1
        result *= 0.0;
    } else {
        result *= this->opts.attractionScale;
    }

    this->state.force[v] += result;
    return lossContribution;
}

double SimpleDotProductEmbedder::repellingForce(NodeId v, NodeId u, VecBuffer<1> &forceBuffer) {
    if (v == u) return 0.0;

    const CVecRef posV = state.currentPositions[v];
    const CVecRef posU = state.currentPositions[u];

    TmpVec<0> result(forceBuffer, 0.0);
    const double dist = vectorOperations::calculateDotProductNorm(posU, posV);

    //displace in random direction if positions are identical
    VecBuffer<2> norms(posV.dimension());
    TmpVec<0> tmpV(norms);
    TmpVec<1> tmpU(norms);
    tmpV = posV.norm() * posV;
    tmpU = posU.norm() * posU;
    bool isSameDirection = true;
    for (int i = 0; i < tmpV.dimension(); i++) {
        if (tmpV[i] != tmpU[i]) {
            isSameDirection = false;
            break;
        }
    }

    if (isSameDirection) {
        std::mt19937 gen = Rand::localGenerator(static_cast<uint32_t>(v), static_cast<uint32_t>(state.currentIteration));
        VecBuffer<1> displace(posU.dimension());
        TmpVec<0> tmpDis(displace);
        tmpDis.setToRandomUnitVector(gen);
        result = posV + 0.05 * tmpDis;
        this->state.force[v] += result;
        return 0.0;
    }

    vectorOperations::differentiateDotProductNorm(posV, posU, dist, result);

    double lossContribution = 0.0;
    if (dist < this->opts.edgeLength) {
        result *= 0;
    } else {
        result *= this->opts.repulsionScale;
        lossContribution = this->opts.edgeLength - dist;
    }

    // increase repulsion force when we use less negative samples
    if (this->opts.numNegativeSamples > 0) {
        result *= static_cast<double>(graphSize()) / static_cast<double>(this->opts.numNegativeSamples);
    }

    this->state.force[v] += result;
    return lossContribution;
}

void SimpleDotProductEmbedder::applyGravityCentre() {
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

void SimpleDotProductEmbedder::observeDisplacement() {
    //TODO: Verify this works with the distance function
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


// ======================================================================================
//
//                       PUBLIC FUNCTIONS SimpleDotProductEmbedder
//
// ======================================================================================


void SimpleDotProductEmbedder::calculateStep() {
    //Increase current step
    state.nextStep();

    //Abort in the case of the first hierarchy layer
    if (graphSize() <= 1) {
        return;
    }

    //Snapshot the positions so we can measure how far the nodes move this step.
    //state.currentPositions still holds the (recentred) positions from the previous step.
    this->previousPositions = this->state.currentPositions;

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
    LOG_WARNING("Dot product embeddings ignore weights")
    this->state.currentWeights = weights;
    sortNodes();
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
