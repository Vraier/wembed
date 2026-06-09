#include <fstream>

#include "NewWEmbedEmbedder.hpp"
#include "VectorOperations.hpp"
#include "WeightedIndex.hpp"


// ======================================================================================
//
//                       PUBLIC FUNCTIONS NewWEmbedEmbedder
//
// ======================================================================================
void NewWEmbedEmbedder::calculateStep() {
    //Increase current step
    params.nextStep();

    //Dump weights to debug file
    if (this->opts.dumpWeights) {
        debug_dumpWeights();
    }

    //Abort in the case of the first hierarchy layer
    if (graphSize() <= 1) {
        this->params.insignificantPosChange = true;
        return;
    }

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

    //Update positions
    this->timer->startTiming("apply_forces", "Applying Forces");
    this->posOptimizer.update(this->currentPositions, this->params.force);
    this->timer->stopTiming("apply_forces");

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

bool NewWEmbedEmbedder::isFinished() {
    return this->params.currentIteration >= this->opts.maxIterations || this->params.insignificantPosChange;
}

void NewWEmbedEmbedder::calculateEmbedding() {
    LOG_INFO("Calculating embedding...");
    timer->startTiming("embedding_all", "Embedding");
    this->params.currentIteration = 0;
    while (!isFinished()) {
        calculateStep();
    }
    timer->stopTiming("embedding_all");
    LOG_INFO("Finished calculating embedding in iteration " << this->params.currentIteration);
}

Graph NewWEmbedEmbedder::getCurrentGraph() {
     return this->graph;
}

std::vector<std::vector<double> > NewWEmbedEmbedder::getCoordinates() {
    return this->currentPositions.convertToVector();
}

std::vector<double> NewWEmbedEmbedder::getWeights() {
    return this->currentWeights;
}

std::vector<util::TimingResult> NewWEmbedEmbedder::getTimings() {
    return timer->getHierarchicalTimingResults();
}

void NewWEmbedEmbedder::setCoordinates(const std::vector<std::vector<double> > &coordinates) {
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

void NewWEmbedEmbedder::setWeights(const std::vector<double> &weights) {
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
//                       PRIVATE FUNCTIONS NewWEmbedEmbedder
//
// ======================================================================================


void NewWEmbedEmbedder::debug_dumpWeights() const {
    const std::string outFile = "weight_dump.txt";
    std::ios_base::openmode mode = std::ios_base::out;
    std::ofstream dumpFile;

    if (this->params.currentIteration <= 1) {
        mode |= std::ios_base::trunc;
    } else {
        mode |= std::ios_base::app;
    }

    dumpFile.open(outFile, mode);
    if (dumpFile.rdstate() == std::fstream::failbit) {
        LOG_ERROR("Trying to open the weight_dump logfile failed. No weights were dumped")
    }

    for (size_t i = 0; i < graphSize(); i++) {
        dumpFile << this->currentWeights[i] << " ";
    }
    dumpFile << std::endl;

    dumpFile.close();
    if (dumpFile.rdstate() == std::fstream::failbit) {
        LOG_ERROR("Trying to close the weight_dump logfile failed, but weights were dumped anyway");
    }
}

void NewWEmbedEmbedder::attractionForce(const NodeId v, const NodeId u, VecBuffer<1> &buffer) {
    if (v == u) return;

    const CVecRef posV = currentPositions[v];
    const CVecRef posU = currentPositions[u];

    TmpVec<0> result(buffer, 0.0);
    const double dist = vectorOperations::calculateLPNorm(posU, posV);

    //displace in random direction if positions are identical
    if (dist <= 0) {
        result.setToRandomUnitVector();
        this->params.force[v] += result;
        return;
    }
    vectorOperations::differentiateLPNormDifference(posU, posV, dist, result);

    const double weightScaling = this->opts.additiveWeights ?
                           (invExpWeights[v] + invExpWeights[u]) :
                           (invExpWeights[v] * invExpWeights[u]);

    if (dist * weightScaling <= this->opts.edgeLength) {
        result *= 0;
    } else {
        result *= this->opts.attractionScale * weightScaling;
    }

    this->params.force[v] += result;
}

void NewWEmbedEmbedder::repellingForce(const NodeId v, const NodeId u, VecBuffer<1> forceBuffer) {
    if (v == u) return;

    const CVecRef posV = currentPositions[v];
    const CVecRef posU = currentPositions[u];
    TmpVec<0> result(forceBuffer, 0.0);
    const double dist = vectorOperations::calculateLPNorm(posV, posU);

    // displace in random direction if positions are identical
    if (dist <= 0) {
        result.setToRandomUnitVector();
        this->params.force[v] += result;
        return;
    }

    vectorOperations::differentiateLPNormDifference(posV, posU, dist, result);

    // calculate weighted distance
    const double weightScaling = this->opts.additiveWeights ? (invExpWeights[v] + invExpWeights[u])
                                                            : (invExpWeights[v] * invExpWeights[u]);
    if (dist * weightScaling > this->opts.edgeLength) {
        result *= 0;
    } else {
        result *= this->opts.repulsionScale * weightScaling;
    }

    // increase repulsion force when we use less negative samples
    if (this->opts.numNegativeSamples > 0) {
        result *= static_cast<double>(graphSize()) / static_cast<double>(this->opts.numNegativeSamples);
    }

    this->params.force[v] += result;
}

void NewWEmbedEmbedder::updateIndex() {
    if (this->opts.numNegativeSamples >= 0) {
        return; //we are not using a geometric index
    }

    //calculate new indices
    if (this->opts.IndexSize >= 1.0) {
        params.indexToGraphMap.resize(graphSize());
        std::iota(params.indexToGraphMap.begin(), params.indexToGraphMap.end(), 0);
        const std::vector<double> weightBuckets =
            WeightedIndex::getDoublingWeightBuckets(this->currentWeights, this->opts.doublingFactor);
        params.currentWeightedIndex.updateIndices(this->currentPositions, this->currentWeights, weightBuckets);
    } else {
        //Only insert a fraction of nodes into the index
        const int32_t numNodes = std::max(1, static_cast<int32_t>(graphSize() * this->opts.IndexSize));
        params.indexToGraphMap = Rand::randomSample(static_cast<int>(graphSize()), numNodes);
        VecList positions(this->opts.embeddingDimension, numNodes);
        std::vector<double> weights(numNodes);

#pragma omp parallel for default(none) shared(numNodes, positions, weights, params) schedule(static)
        for (size_t i = 0; i < numNodes; i++) {
            positions[i] = this->currentPositions[params.indexToGraphMap[i]];
            weights[i] = this->currentWeights[params.indexToGraphMap[i]];
        }

        const std::vector<double> weightBuckets = WeightedIndex::getDoublingWeightBuckets(weights, this->opts.doublingFactor);
        params.currentWeightedIndex.updateIndices(positions, weights, weightBuckets);
    }
}

std::vector<NodeId> NewWEmbedEmbedder::getRepellingCandidatesForNode(NodeId v, VecBuffer<2> &buffer) const {
    //TODO: Definitely think about refactoring this
    std::vector<NodeId> candidates;

    if (this->opts.numNegativeSamples >= 0) {
        candidates = sampleRandomNoise(std::min(static_cast<int32_t>(graphSize()), this->opts.numNegativeSamples));
        return candidates;
    }

    this->params.currentWeightedIndex.getNodesWithinWeightedDistance(this->currentPositions[v], this->currentWeights[v], this->opts.edgeLength,
                                                       candidates, buffer);
    for (NodeId& candidate: candidates) {
        candidate = this->params.indexToGraphMap[candidate];
        ASSERT(candidate < graphSize() && candidate >= 0, "Index out of bounds: " << candidate << " for N = " << graphSize());
    }
    return candidates;
}

void NewWEmbedEmbedder::calculateAllAttractingForces() {
    VecBuffer<1> buffer(this->opts.embeddingDimension);
#pragma omp parallel for default(none) firstprivate(buffer) shared(sortedNodeIDs, graph) schedule(runtime)
    for (const NodeId v : this->sortedNodeIDs) {
        for (const NodeId u : graph.getNeighbors(v)) {
            attractionForce(v, u, buffer);
        }
    }
}

void NewWEmbedEmbedder::calculateAllRepellingForces() {
    VecBuffer<2> indexBuffer(this->opts.embeddingDimension);
    VecBuffer<1> forceBuffer(this->opts.embeddingDimension);
    numRepForceCalculations = 0;

#pragma omp parallel for default(none) firstprivate(indexBuffer, forceBuffer), reduction(+:numRepForceCalculations), schedule(runtime)
    for (const NodeId v : sortedNodeIDs) {
        const std::vector<NodeId> repellingCandidates = getRepellingCandidatesForNode(v, indexBuffer);
        for (const NodeId u : repellingCandidates) {
            if (graph.areNeighbors(v, u) || graph.areInSameColorClass(v, u)) {
                continue;
            }
            repellingForce(v, u, forceBuffer);
            numRepForceCalculations++;
        }
    }
}

//TODO: This could be moved somewhere else
std::vector<NodeId> NewWEmbedEmbedder::sampleRandomNoise(const int32_t numNodes) const {
    return Rand::randomSample(static_cast<int32_t>(graphSize()), numNodes);
}

std::vector<double> NewWEmbedEmbedder::rescaleWeights() const {
    const uint32_t N = graphSize();
    std::vector<double> rescaledWeights = constructDegreeWeights();

    for (NodeId v = 0; v < N; v++) {
        if (this->opts.dimensionHint > 0) {
            rescaledWeights[v] = Toolkit::myPow(rescaledWeights[v],
                static_cast<double>(this->opts.embeddingDimension) / static_cast<double>(this->opts.dimensionHint));
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

std::vector<double> NewWEmbedEmbedder::constructDegreeWeights() const {
    std::vector<double> weights(graphSize());
    for (NodeId v = 0; v < graphSize(); v++) {
        weights[v] = this->graph.getNumNeighbors(v);
    }
    return weights;
}

std::vector<double> NewWEmbedEmbedder::constructUnitWeights() const {
    return std::vector(graphSize(), 1.0);
}