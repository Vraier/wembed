#include <fstream>
#include <thread>


#include "WembedEmbedder.hpp"

#include "ParallelReduce.hpp"
#include "VectorOperations.hpp"


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

    const double lossContribution = dist - this->opts.edgeLength / weightScaling;
    if (dist * weightScaling <= this->opts.edgeLength) {
        result *= this->opts.repulsionScale * weightScaling; //Attract to counter repulsion force
    } else {
        result *= this->opts.attractionScale * weightScaling;
    }

    this->state.force[v] += result;
    return lossContribution;
}

double WembedEmbedder::repellingForce(const NodeId v, const NodeId u, TmpVec<0>& result) {
    if (v == u) return 0.0;

    const CVecRef posV = state.currentPositions[v];
    const CVecRef posU = state.currentPositions[u];
    const double dist = vectorOperations::calculateLPNorm(posV, posU);

    // displace in random direction if positions are identical (see attractionForce)
    if (dist <= 0) {
        std::mt19937 gen = Rand::localGenerator(static_cast<uint32_t>(v), static_cast<uint32_t>(state.currentIteration));
        result.setToRandomUnitVector(gen);
        this->state.force[v] +=  result;
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
    return lossContribution;
}


double WembedEmbedder::scatterRepulsion(const NodeId v, const std::vector<NodeId> &candidates, VecList& forces, const size_t threadCount) {
    const size_t tid = omp_get_thread_num();

    VecBuffer<1> forceBuffer(this->opts.embeddingDimension);

    double lossContribution = 0.0;
    for (auto& u : candidates) {
        TmpVec<0> result(forceBuffer, 0.0);
        lossContribution += repellingForce(v, u, result);
        forces[v * threadCount + tid] += result;
        forces[u * threadCount + tid] -= result;
    }
    return lossContribution;
}

void WembedEmbedder::selectNodes(std::vector<CVecRef>& points) {

    if (this->opts.IndexSize >= 1.0) {

        state.indexToGraphMap.resize(graphSize());
        points.resize(graphSize());

#pragma omp parallel for default(none) shared(points, state) schedule(static)
        for (int i = 0; i < graphSize(); i++) {
            this->state.indexToGraphMap[i] = i;
            points[i] = this->state.currentPositions[i];
        }

    } else {

        //Only insert a fraction of nodes into the index
        const int32_t numNodes = std::max(1, static_cast<int32_t>(graphSize() * this->opts.IndexSize));
        state.indexToGraphMap = Rand::randomSample(static_cast<int>(graphSize()), numNodes);
        points.resize(numNodes);

#pragma omp parallel for default(none) shared(numNodes, points, state) schedule(static)
        for (int i = 0; i < numNodes; i++) {
            points[i] = this->state.currentPositions[state.indexToGraphMap[i]];
        }

    }
}

void WembedEmbedder::updateIndex() {
    if (this->opts.numNegativeSamples >= 0) {
        return; //we are not using a geometric index
    }

    std::vector<CVecRef> points;
    selectNodes(points);
    state.currentWeightedIndex.updateIndex(points);
}

std::vector<NodeId> WembedEmbedder::getRepellingCandidatesForNode(NodeId v, [[maybe_unused]] VecBuffer<2> &buffer) const {
    std::vector<NodeId> candidates;

    if (this->opts.numNegativeSamples >= 0) {
        candidates = sampleRandomNoise(std::min(static_cast<int32_t>(graphSize()), this->opts.numNegativeSamples));
        return candidates;
    }

    std::vector<uint64_t> queryResults;
    this->state.currentWeightedIndex.querySphere(this->state.currentPositions[v], this->state.currentWeights[v], this->opts.edgeLength, queryResults);

    if (this->opts.IndexSize < 1.0) {
        for (uint64_t& r: queryResults) {
            r = this->state.indexToGraphMap[r];
            ASSERT(r < graphSize());
        }
    }

    //Filter candidates
    candidates.reserve(queryResults.size());
    for (const uint64_t queryResult : queryResults) {
        const auto u = static_cast<NodeId>(queryResult);
        if (state.currentWeights[v] < state.currentWeights[u]) continue;
        if (state.currentWeights[v] == state.currentWeights[u] && v > u) continue;

        candidates.push_back(u);
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
    const double loss = util::deterministicSum(graphSize(), [this](std::size_t i) { return this->lossPerNode[i]; });
    this->state.lastAttractLoss = loss;
    this->state.lastRepelLoss = loss; //Counter repulsion computation for neighbours
}

void WembedEmbedder::calculateAllRepellingForces() {
    VecBuffer<2> indexBuffer(this->opts.embeddingDimension);
    VecBuffer<1> forceBuffer(this->opts.embeddingDimension);
    numRepForceCalculations = 0;

    //Parallel computation of repulsion forces
    const size_t threadCount = std::thread::hardware_concurrency();
    VecList forces(this->opts.embeddingDimension,graphSize() * threadCount);

#pragma omp parallel for num_threads(threadCount) default(none) shared(indexBuffer, forces, threadCount) reduction(+:numRepForceCalculations) schedule(dynamic)
    for (const NodeId v : state.sortedNodeIDs) {
        const std::vector<NodeId> repellingCandidates = getRepellingCandidatesForNode(v, indexBuffer);
        const double nodeLoss = scatterRepulsion(v, repellingCandidates, forces, threadCount);
        this->lossPerNode[v] = nodeLoss;

        numRepForceCalculations += repellingCandidates.size();
    }

    //Add results into force vector
#pragma omp parallel for num_threads(threadCount) default(none) shared(threadCount, forces) schedule(dynamic)
    for (size_t i = 0; i < graphSize(); i++) {
        for (size_t t = 0; t < threadCount; t++) {
            this->state.force[i] += forces[i * threadCount + t];
        }
    }

    //Addition as we have the neighbor offset
    this->state.lastRepelLoss +=
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