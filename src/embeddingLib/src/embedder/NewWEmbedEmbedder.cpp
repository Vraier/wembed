#include <fstream>

#include "NewWEmbedEmbedder.hpp"
#include "VectorOperations.hpp"


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
    if (currentWeights[v] < currentWeights[u]) return;
    if (currentWeights[v] == currentWeights[u] && v > u) return;

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

    candidateLocks[v].lock();
    this->params.force[v] += result;
    candidateLocks[v].unlock();
    candidateLocks[u].lock();
    this->params.force[u] -= result;
    candidateLocks[u].unlock();
}

void NewWEmbedEmbedder::selectNodes(std::vector<std::pair<CVecRef, NodeId>>& points) {

    if (this->opts.IndexSize >= 1.0) {

        params.indexToGraphMap.resize(graphSize());
        points.resize(graphSize());

//#pragma omp parallel for default(none) shared(points, params) schedule(static)
        for (size_t i = 0; i < graphSize(); i++) {
            this->params.indexToGraphMap[i] = i;
            points[i] = std::make_pair(this->currentPositions[i], i);
        }

    } else {

        //Only insert a fraction of nodes into the index
        const int32_t numNodes = std::max(1, static_cast<int32_t>(graphSize() * this->opts.IndexSize));
        params.indexToGraphMap = Rand::randomSample(static_cast<int>(graphSize()), numNodes);
        points.resize(numNodes);

//#pragma omp parallel for default(none) shared(numNodes, points, params) schedule(static)
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

std::vector<NodeId> NewWEmbedEmbedder::getRepellingCandidatesForNode(NodeId v, [[maybe_unused]] VecBuffer<2> &buffer) const {
    std::vector<NodeId> candidates;

    if (this->opts.numNegativeSamples >= 0) {
        candidates = sampleRandomNoise(std::min(static_cast<int32_t>(graphSize()), this->opts.numNegativeSamples));
        return candidates;
    }

    this->params.weightedIndex.querySphere(this->currentPositions[v], this->currentWeights[v], this->opts.edgeLength, candidates);

    for (NodeId& candidate: candidates) {
        candidate = this->params.indexToGraphMap[candidate];
        ASSERT(candidate < graphSize() && candidate >= 0, "Index out of bounds: " << candidate << " for N = " << graphSize());
    }

    return candidates;
}

static size_t find(std::vector<int>& vec, int value) {
    size_t l = 0;
    size_t r = vec.size();
    while (l < r) {
        size_t currPos = (l + r) / 2;
        if (vec[currPos] < value) {
            l = currPos + 1;
        } else if (vec[currPos] > value) {
            r = currPos;
        } else if (vec[currPos] == value) {
            return currPos;
        }
    }
    return vec.size();
}

static bool candidateVerification(std::vector<std::vector<int>> candidates) {
    for (auto & candidate : candidates) {
        std::ranges::sort(candidate);
    }
    bool valid = true;
    //Check for duplicates
    for (size_t i = 0; i < candidates.size(); i++) {
        for (size_t j = 1; j < candidates[i].size(); j++) {
            if (candidates[i][j - 1] == candidates[i][j]) {
                LOG_WARNING("There is a duplicate in the candidates of node " +
                            std::to_string(i) + ". Node " +
                            std::to_string(j) + " appears twice");
                valid = false;
            }
        }
    }

    //Check for symmetry
    for (size_t i = 0; i < candidates.size(); i++) {
        for (size_t j = 0; j < candidates[i].size(); j++) {
            if (candidates[i].size() == find(candidates[candidates[i][j]], i)) {
                LOG_WARNING("The candidates for " + std::to_string(i) + " contain " + std::to_string(j)
                            + ", but not vice versa");
                valid = false;
            }
        }
    }

    return valid;
}

std::vector<std::vector<NodeId> > NewWEmbedEmbedder::getAllRepellingCandidates() {
    if constexpr (true) {
        std::vector<std::vector<NodeId>> candidates(graphSize());
        VecBuffer<2> indexBuffer(this->opts.embeddingDimension);
        //Stores the sizes of each candidate vector after node removal
        //This allows to ignore all nodes that might be added to the vector by a different thread
        std::vector<size_t> candidateSizes(graphSize());

        //Get candidates for each Node
        timer->startTiming("CandidateSearch", "Searching for repelling candidates");
#pragma omp parallel for firstprivate(indexBuffer) shared(candidates) schedule(dynamic)
        for (size_t v = 0; v < graphSize(); v++) {
            const std::vector<NodeId> tmp = getRepellingCandidatesForNode(sortedNodeIDs[v], indexBuffer);
            candidates[v] = tmp;
        }
        timer->stopTiming("CandidateSearch");

        //Remove elements that are smaller and compute candidate sizes
        timer->startTiming("NodeRemoval", "Remove heavier nodes");
#pragma omp parallel for default(none) shared(candidates, candidateSizes) schedule(dynamic)
        for (NodeId v = 0; v < graphSize(); v++) {
            for (NodeId u = 0; u < candidates[v].size(); u++) {
                if (currentWeights[candidates[v][u]] > currentWeights[v]) {
                    candidates[v].erase(candidates[v].begin() + u);
                }
                else if (currentWeights[candidates[v][u]] == currentWeights[v] && candidates[v][u] > v) {
                    candidates[v].erase(candidates[v].begin() + u);
                }
            }
            candidateSizes[v] = candidates[v].size();
        }
        timer->stopTiming("NodeRemoval");

        //Make things symmetric:
        timer->startTiming("Symmetrizising", "Making the repelling candidates symmetric");
#pragma omp parallel for default(none) shared(candidates, candidateSizes) schedule(dynamic)
        for (int v = 0; v < graphSize(); v++) {
            for (size_t u = 0; u < candidateSizes[v]; u++) {
                candidateLocks[u].lock();
                candidates[u].push_back(v);
                candidateLocks[u].unlock();
            }
        }
        timer->stopTiming("Symmetrizising");
        //Verifying the results could be correct
        if constexpr (false && !candidateVerification(candidates)) {
            LOG_WARNING("No symmetric candidates");
        }
        return candidates;
    } else { //TODO: NOPE, F1 score is 0.46
        std::vector<std::vector<NodeId>> candidates(graphSize());
        VecBuffer<2> indexBuffer(this->opts.embeddingDimension);

        timer->startTiming("CandidateSearch", "Searching for repelling candidates");
#pragma omp parallel for default(none) shared(candidates, indexBuffer) schedule(dynamic)
        for (NodeId v = 0; v < graphSize(); v++) {
            const std::vector<NodeId> repellingCandidates = getRepellingCandidatesForNode(sortedNodeIDs[v], indexBuffer);

            this->candidateLocks[v].lock();
            candidates[v].insert(candidates[v].end(), repellingCandidates.begin(), repellingCandidates.end());
            this->candidateLocks[v].unlock();

            for (const int repellingCandidate : repellingCandidates) {
                if (currentWeights[repellingCandidate] < currentWeights[v]) {
                    candidateLocks[repellingCandidate].lock();
                    candidates[repellingCandidate].push_back(v);
                    candidateLocks[repellingCandidate].unlock();
                }
            }
        }
        timer->stopTiming("CandidateSearch");
        //Verifying the results could be correct
        if constexpr (false && !candidateVerification(candidates)) {
            LOG_WARNING("No symmetric candidates");
        }
        return candidates;
    }
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

std::vector<double> NewWEmbedEmbedder::rescaleWeights(const double dimensionHint, const double embeddingDimension,
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

std::vector<double> NewWEmbedEmbedder::constructDegreeWeights(const Graph& g) {
    std::vector<double> weights(g.getNumVertices());
    for (NodeId v = 0; v < g.getNumVertices(); v++) {
        weights[v] = g.getNumNeighbors(v);
    }
    return weights;
}

std::vector<double> NewWEmbedEmbedder::constructUnitWeights(const int N) {
    std::vector<double> weights(N);
    for (NodeId v = 0; v < N; v++) {
        weights[v] = 1.0;
    }
    return weights;
}