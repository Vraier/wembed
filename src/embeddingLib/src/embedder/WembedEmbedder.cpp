#include <fstream>
#include <thread>

#include "WembedEmbedder.hpp"
#include "VectorOperations.hpp"


// ======================================================================================
//
//                       PUBLIC FUNCTIONS WembedEmbedder
//
// ======================================================================================
void WembedEmbedder::calculateStep() {

    //Increase current step
    params.nextStep();

    //Abort in the case of the first hierarchy layer
    if (graphSize() <= 1) {
        this->params.insignificantPosChange = true;
        return;
    }

    //TODO: optimize storing of old positions (Implement std::move for VecLists)
    VecList oldPositions(this->currentPositions.dimension(), this->currentPositions.size());
#pragma omp parallel for default(none) shared(oldPositions) schedule(static)
    for (size_t i = 0; i < graphSize(); i++) {
        oldPositions[i] = this->currentPositions[i];
    }

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
    this->posOptimizer->update(this->currentPositions, this->params.force);
    this->timer->stopTiming("apply_forces");

    this->timer->startTiming("gravity", "Move graph towards centre");
    applyGravityCentre();
    this->timer->stopTiming("gravity");

    //calculate change in positions
    this->timer->startTiming("position_change", "Change in Positions");
    VecBuffer<1> buffer(this->opts.embeddingDimension);
    double sumNormDiffSquared = 0.0;

#pragma omp parallel for default(none) firstprivate(buffer) shared(oldPositions, currentPositions) reduction(+:sumNormDiffSquared) schedule(static)
    for (size_t v = 0; v < graphSize(); v++) {
        TmpVec<0> tmpVec(buffer);
        tmpVec = oldPositions[v] - currentPositions[v];
        sumNormDiffSquared += tmpVec.sqNorm();
    }

    const double averageNormDiff = sumNormDiffSquared / graphSize();

    if (this->params.currentIteration == 1 || (this->params.currentIteration > 0 && this->params.currentIteration % 10 == 0)) {
        std::cout << "(Iteration " << this->params.currentIteration << ": #rep forces " << numRepForceCalculations
                  << ", relative pos change: " << averageNormDiff << ")" << std::endl;
    }

    if (averageNormDiff < this->opts.positionMinChange) {
        this->params.insignificantPosChange = true;
    }

    this->timer->stopTiming("position_change");
}

bool WembedEmbedder::isFinished() {
    return this->params.currentIteration >= this->opts.maxIterations || this->params.insignificantPosChange;
}

void WembedEmbedder::calculateEmbedding() {
    LOG_INFO("Calculating embedding...");
    timer->startTiming("embedding_all", "Embedding");
    this->params.currentIteration = 0;
    while (!isFinished()) {
        calculateStep();
    }
    timer->stopTiming("embedding_all");
    LOG_INFO("Finished calculating embedding in iteration " << this->params.currentIteration);
}

Graph WembedEmbedder::getCurrentGraph() {
     return this->graph;
}

std::vector<std::vector<double> > WembedEmbedder::getCoordinates() {
    return this->currentPositions.convertToVector();
}

std::vector<double> WembedEmbedder::getWeights() {
    return this->currentWeights;
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
            currentPositions[i][d] = coordinates[i][d];
        }
    }
}

void WembedEmbedder::setWeights(const std::vector<double> &weights) {
    ASSERT(graphSize() == weights.size());

    this->currentWeights = weights;
    sortNodes();

#pragma omp parallel for default(none) shared(invExpWeights, currentWeights) schedule(static)
    for (size_t i = 0; i < graphSize(); i++) {
        invExpWeights[i] = 1.0 / Toolkit::myPow(currentWeights[i], 1.0 / static_cast<double>(opts.embeddingDimension));
    }
}

// ======================================================================================
//
//                       PRIVATE FUNCTIONS WembedEmbedder
//
// ======================================================================================


double WembedEmbedder::attractionForce(const NodeId v, const NodeId u, VecBuffer<1>& forceBuffer) {
    if (v == u) return 0.0;

    const CVecRef posV = currentPositions[v];
    const CVecRef posU = currentPositions[u];

    TmpVec<0> result(forceBuffer, 0.0);
    const double dist = vectorOperations::calculateLPNorm(posU, posV);

    //displace in random direction if positions are identical
    if (dist <= 0) {
        result.setToRandomUnitVector();
        this->params.force[v] += result;
        return 0.0;
    }
    vectorOperations::differentiateLPNormDifference(posU, posV, dist, result);

    const double weightScaling = this->opts.additiveWeights ?
                           (invExpWeights[v] + invExpWeights[u]) :
                           (invExpWeights[v] * invExpWeights[u]);

    double lossContribution = 0.0;
    if (dist * weightScaling <= this->opts.edgeLength) {
        result *= this->opts.repulsionScale * weightScaling; //Attract to counter repulsion force
        //TODO: Do I need loss contribution here as well?
    } else {
        result *= this->opts.attractionScale * weightScaling;
        lossContribution = dist - this->opts.edgeLength / weightScaling;
    }

    this->params.force[v] += result;
    return lossContribution;
}

double WembedEmbedder::repellingForce(const NodeId v, const NodeId u, VecBuffer<1>& forceBuffer) {
    if (v == u) return 0.0;

    const CVecRef posV = currentPositions[v];
    const CVecRef posU = currentPositions[u];
    TmpVec<0> result(forceBuffer, 0.0);
    const double dist = vectorOperations::calculateLPNorm(posV, posU);

    // displace in random direction if positions are identical
    if (dist <= 0) {
        result.setToRandomUnitVector();
        this->params.force[v] += result;
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


void NewWEmbedEmbedder::scatterRepulsion(const NodeId v, const std::vector<NodeId> &candidates, VecList& forces, const size_t threadCount) {
    const size_t tid = omp_get_thread_num();

    VecBuffer<1> forceBuffer(this->opts.embeddingDimension);

    for (auto& u : candidates) {
        TmpVec<0> result(forceBuffer, 0.0);
        repellingForce(v, u, result);
        forces[v * threadCount + tid] += result;
        forces[u * threadCount + tid] -= result;
    }
}

void NewWEmbedEmbedder::selectNodes(std::vector<std::pair<CVecRef, NodeId>>& points) {

    if (this->opts.IndexSize >= 1.0) {

        params.indexToGraphMap.resize(graphSize());
        points.resize(graphSize());

#pragma omp parallel for default(none) shared(points, params) schedule(static)
        for (int i = 0; i < graphSize(); i++) {
            this->params.indexToGraphMap[i] = i;
            points[i] = std::make_pair(this->currentPositions[i], i);
        }

    } else {

        //Only insert a fraction of nodes into the index
        const int32_t numNodes = std::max(1, static_cast<int32_t>(graphSize() * this->opts.IndexSize));
        params.indexToGraphMap = Rand::randomSample(static_cast<int>(graphSize()), numNodes);
        points.resize(numNodes);

#pragma omp parallel for default(none) shared(numNodes, points, params) schedule(static)
        for (int i = 0; i < numNodes; i++) {
            points[i] = std::make_pair(this->currentPositions[params.indexToGraphMap[i]], i);
        }

    }
}

void NewWEmbedEmbedder::updateIndex() {
    if (this->opts.numNegativeSamples >= 0) {
        return; //we are not using a geometric index
    }

    std::vector<std::pair<CVecRef, NodeId>> points;
    selectNodes(points);
    params.weightedIndex.updateIndex(points);
}

std::vector<NodeId> WembedEmbedder::getRepellingCandidatesForNode(NodeId v, [[maybe_unused]] VecBuffer<2> &buffer) const {
    std::vector<NodeId> candidates;

    if (this->opts.numNegativeSamples >= 0) {
        candidates = sampleRandomNoise(std::min(static_cast<int32_t>(graphSize()), this->opts.numNegativeSamples));
        return candidates;
    }

    std::vector<uint64_t> queryResults;
    this->params.weightedIndex.querySphere(this->currentPositions[v], this->currentWeights[v], this->opts.edgeLength, queryResults);

    if (this->opts.IndexSize < 1.0) {
        for (uint64_t& r: queryResults) {
            r = this->params.indexToGraphMap[r];
            ASSERT(r < graphSize());
        }
    }

    //Filter candidates
    candidates.reserve(queryResults.size());
    for (const uint64_t queryResult : queryResults) {
        const auto u = static_cast<NodeId>(queryResult);
        if (currentWeights[v] < currentWeights[u]) continue;
        if (currentWeights[v] == currentWeights[u] && v > u) continue;

        candidates.push_back(u);
    }

    return candidates;
}

void WembedEmbedder::calculateAllAttractingForces() {
    VecBuffer<1> buffer(this->opts.embeddingDimension);
    double attractLoss = 0.0;
#pragma omp parallel for default(none) firstprivate(buffer) shared(sortedNodeIDs, graph) reduction(+:attractLoss) schedule(runtime)
    for (const NodeId v : this->sortedNodeIDs) {
        for (const NodeId u : graph.getNeighbors(v)) {
            attractLoss += attractionForce(v, u, buffer);
        }
    }
    this->params.lastAttractLoss = attractLoss;
}

void WembedEmbedder::calculateAllRepellingForces() {
    VecBuffer<2> indexBuffer(this->opts.embeddingDimension);
    VecBuffer<1> forceBuffer(this->opts.embeddingDimension);
    numRepForceCalculations = 0;
    double repelLoss = 0.0;

    //Parallel computation of repulsion forces
    const size_t threadCount = std::thread::hardware_concurrency();
    VecList forces(this->opts.embeddingDimension,graphSize() * threadCount);

#pragma omp parallel for num_threads(threadCount) default(none) shared(indexBuffer, forces, threadCount) reduction(+:numRepForceCalculations) schedule(dynamic)
    for (const NodeId v : sortedNodeIDs) {
        const std::vector<NodeId> repellingCandidates = getRepellingCandidatesForNode(v, indexBuffer);
        scatterRepulsion(v, repellingCandidates, forces, threadCount);
        numRepForceCalculations += repellingCandidates.size();
    }

    //Add results into force vector
#pragma omp parallel for num_threads(threadCount) default(none) shared(threadCount, forces) schedule(dynamic)
    for (size_t i = 0; i < graphSize(); i++) {
        for (size_t t = 0; t < threadCount; t++) {
            this->params.force[i] += forces[i * threadCount + t];
        }
    }
}

void WembedEmbedder::calculateAllCentreForces() {
#pragma omp parallel for default(none) shared(sortedNodeIDs, opts, params, currentPositions) schedule(static)
    for (const NodeId v : this->sortedNodeIDs) {
        this->params.force[v] += -1.0 * this->opts.centreScale * this->currentPositions[v];
    }
}

void WembedEmbedder::applyGravityCentre() {
    std::vector<double> dimGravity(this->opts.embeddingDimension);
    for (int dim = 0; dim < this->opts.embeddingDimension; dim++) {
        double dimensionSum = 0.0;
#pragma omp parallel for default(none) shared(dim) reduction(+:dimensionSum) schedule(static)
        for (size_t v = 0; v < graphSize(); v++) {
            dimensionSum += this->currentPositions[v][dim];
        }
        dimGravity[dim] = dimensionSum / static_cast<double>(graphSize());
    }
    // Wrap the centroid in a single-row VecList so we can use VecRef arithmetic below.
    // TODO: this can be a temp vec probably
    VecList gravityCentre({dimGravity});

#pragma omp parallel for default(none) shared(gravityCentre) schedule(static)
    for (size_t i = 0; i < graphSize(); i++) {
        this->currentPositions[i] -= gravityCentre[0];
    }
}

void WembedEmbedder::applyGravityCentre() {
    std::vector<double> dimGravity(this->opts.embeddingDimension);
    for (int dim = 0; dim < this->opts.embeddingDimension; dim++) {
        double dimensionSum = 0.0;
#pragma omp parallel for default(none) shared(dim) reduction(+:dimensionSum) schedule(static)
        for (size_t v = 0; v < graphSize(); v++) {
            dimensionSum += this->currentPositions[v][dim];
        }
        dimGravity[dim] = dimensionSum / static_cast<double>(graphSize());
    }
    // Wrap the centroid in a single-row VecList so we can use VecRef arithmetic below.
    // TODO: this can be a temp vec probably
    VecList gravityCentre({dimGravity});

#pragma omp parallel for default(none) shared(gravityCentre) schedule(static)
    for (size_t i = 0; i < graphSize(); i++) {
        this->currentPositions[i] -= gravityCentre[0];
    }
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