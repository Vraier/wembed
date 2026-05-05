#include <execution>
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
    this->currentIteration++;

    //Dump weights to debug file
    if (this->opts.dumpWeights) {
        debug_dumpWeights();
    }

    //Abort in the case of the first hierarchy layer
    if (graphSize() <= 1) {
        this->insignificantPosChange = true;
        return;
    }

    // Declare and define all temporary containers and parameters
    VecList force(this->opts.embeddingDimension, graphSize());
    std::vector<double> weightParameterForce(graphSize(), 0);
    std::vector<NodeId> indexToGraphMap;
    WeightedIndex currentWeightedIndex(this->opts.indexType, this->opts.embeddingDimension);

    VecList oldPositions(this->currentPositions.dimension(), this->currentPositions.size());
#pragma omp parallel for default(none) shared(oldPositions) schedule(static)
    for (size_t i = 0; i < graphSize(); i++) {
        oldPositions[i] = this->currentPositions[i];
    }

    //Rebuild indices
    this->timer->startTiming("index", "Construct spacial index");
    updateIndex(indexToGraphMap, currentWeightedIndex);
    this->timer->stopTiming("index");

    //Compute attracting forces
    this->timer->startTiming("attracting_forces", "Compute Attracting Forces");
    calculateAllAttractingForces(force, weightParameterForce);
    this->timer->stopTiming("attracting_forces");

    //Compute repelling forces
    this->timer->startTiming("repelling_forces", "Compute Repelling Forces");
    calculateAllRepellingForces(currentWeightedIndex, indexToGraphMap);
    this->timer->stopTiming("repelling_forces");

    //TODO: Refactor from here
    //applying gradient
    this->timer->startTiming("apply_forces", "Applying Forces");
    //Update positions
    this->posOptimizer.update(this->currentPositions, force);
    //Update weights
    if (this->opts.weightLearningRate > 0.0) {
        this->weightOptimizer.update(currentWeightParameters, weightParameterForce);
        for (NodeId i = 0; i < graphSize(); i++) {
            currentWeights[i] = std::log(1 + std::exp(this->currentWeightParameters[i]));
        }
    }
    this->timer->stopTiming("apply_forces");

    //calculate change in positions
    this->timer->startTiming("position_change", "Change in Positions");
    VecBuffer<1> buffer(this->opts.embeddingDimension);
    double sumNormDiffSquared = 0.0;

    //TODO: parallel
    for (size_t v = 0; v < graphSize(); v++) {
        TmpVec<0> tmpVec(buffer);
        tmpVec = oldPositions[v] - currentPositions[v];
        sumNormDiffSquared += tmpVec.sqNorm();
    }
    const double averageNormDiff = sumNormDiffSquared / graphSize();

    if (this->currentIteration == 1 || (currentIteration > 0 && currentIteration % 10 == 0)) {
        std::cout << "(Iteration " << currentIteration << ": #rep forces " << numRepForceCalculations
                  << ", relative pos change: " << averageNormDiff << ")" << std::endl;
    }

    if (averageNormDiff < this->opts.positionMinChange) {
        this->insignificantPosChange = true;
    }

    this->timer->stopTiming("position_change");
}

bool NewWEmbedEmbedder::isFinished() {
    return this->currentIteration >= this->opts.maxIterations || this->insignificantPosChange;
}

void NewWEmbedEmbedder::calculateEmbedding() {
    LOG_INFO("Calculating embedding...");
    timer->startTiming("embedding_all", "Embedding");
    currentIteration = 0;
    while (!isFinished()) {
        calculateStep();
    }
    timer->stopTiming("embedding_all");
    LOG_INFO("Finished calculating embedding in iteration " << currentIteration);
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
    computeWeightPrefixSum();

    //TODO: parallel?
    for (size_t i = 0; i < graphSize(); i++) {
        currentWeightParameters[i] = std::log(std::exp(currentWeights[i]) - 1.0);
    }
}

// ======================================================================================
//
//                       PRIVATE FUNCTIONS NewWEmbedEmbedder
//
// ======================================================================================

[[nodiscard]] constexpr uint32_t NewWEmbedEmbedder::graphSize() const {
    return this->graph.getNumVertices();
}

void NewWEmbedEmbedder::computeWeightPrefixSum() {
    //TODO: parallel?
    weightPrefixSum[0] = currentWeights[0];
    for (size_t i = 1; i < currentWeights.size(); i++) {
        weightPrefixSum[i] = currentWeights[i] + weightPrefixSum[i - 1];
    }
}

void NewWEmbedEmbedder::sortNodes() {
    std::iota(sortedNodeIDs.begin(), sortedNodeIDs.end(), 0);
    std::sort(std::execution::par_unseq, sortedNodeIDs.begin(), sortedNodeIDs.end(),
              [this](const int a , const int b) -> bool {return this->currentWeights[a] > this->currentWeights[b];});
}

void NewWEmbedEmbedder::attractionForce(const NodeId v, const NodeId u, VecList& force, VecBuffer<1> &buffer) {
    //TODO: maybe refactor
    if (v == u) return;

    const CVecRef posV = currentPositions[v];
    const CVecRef posU = currentPositions[u];

    TmpVec<0> result(buffer, 0.0);
    const double dist = vectorOperations::calculateLPNorm(posU, posV, this->opts.lpNorm);

    //displace in random direction if positions are identical
    if (dist <= 0) {
        result.setToRandomUnitVector();
        force[v] += result;
        return;
    }
    vectorOperations::differentiateLPNormDifference(posU, posV, result, this->opts.lpNorm);

    const double wv = currentWeights[v];
    const double wu = currentWeights[u];
    const double weightScaling = this->opts.additiveWeights ?
                           (Toolkit::myPow(wv, 1.0 / this->opts.embeddingDimension) +
                           Toolkit::myPow(wu, 1.0 / this->opts.embeddingDimension)) :
                           Toolkit::myPow(wu * wv, 1.0 / this->opts.embeddingDimension);

    const double weightDist = dist / weightScaling;

    if (weightDist <= this->opts.edgeLength) {
        result *= 0;
    } else {
        result *= this->opts.attractionScale / weightScaling;
    }

    force[v] += result;
}

void NewWEmbedEmbedder::attractionWeightForce(const NodeId v, const NodeId u, std::vector<double> &weightParameterForce, VecBuffer<1> &buffer) {
    //TODO: maybe refactor
    if (this->opts.weightLearningRate <= 0.0 || v == u) return;

    const CVecRef posV = this->currentPositions[v];
    const CVecRef posU = this->currentPositions[u];

    const double wv = currentWeights[v];
    const double wu = currentWeights[u];
    const double hiddenParameter = this->currentWeightParameters[v];
    TmpVec<0> tmp(buffer, 0.0);
    tmp = posV - posU;
    const double dist = tmp.norm();
    const double weightDist = dist / Toolkit::myPow(wu * wv, 1.0 / this->opts.embeddingDimension);

    if (weightDist <= this->opts.edgeLength) return;

    const double exPlus1 = std::exp(hiddenParameter) + 1;
    const double result = dist * wu * std::exp(hiddenParameter) /
                          static_cast<double>(this->opts.embeddingDimension) * exPlus1 *
                          Toolkit::myPow(wu * std::log(exPlus1),
                                         static_cast<double>(this->opts.embeddingDimension + 1) /
                                             static_cast<double>(this->opts.embeddingDimension));

    weightParameterForce[v] += result;
}

void NewWEmbedEmbedder::repellingForce(const NodeId v, const NodeId u, VecBuffer<1> forceBuffer) {
    //TODO:
}

void NewWEmbedEmbedder::repellingWeightForce(const NodeId v, const NodeId u, VecBuffer<1> forceBuffer) {
    //TODO:
}

void NewWEmbedEmbedder::debug_dumpWeights() const {
    const std::string outFile = "weight_dump.txt";
    std::ios_base::openmode mode = std::ios_base::out;
    std::ofstream dumpFile;

    if (currentIteration <= 1) {
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

void NewWEmbedEmbedder::updateIndex(std::vector<NodeId> &indexToGraphMap, WeightedIndex &currentWeightedIndex) {
    if (this->opts.numNegativeSamples >= 0) {
        return; //we are not using a geometric index
    }

    //calculate new indices
    if (this->opts.IndexSize >= 1.0) {
        indexToGraphMap.resize(graphSize());
        std::iota(indexToGraphMap.begin(), indexToGraphMap.end(), 0);
        const std::vector<double> weightBuckets =
            WeightedIndex::getDoublingWeightBuckets(this->currentWeights, this->opts.doublingFactor);
        currentWeightedIndex.updateIndices(this->currentPositions, this->currentWeights, weightBuckets);
    } else {
        //Only insert a fraction of nodes into the index
        const int32_t numNodes = std::max(1, static_cast<int32_t>(graphSize() * this->opts.IndexSize));
        indexToGraphMap = Rand::randomSample(static_cast<int>(graphSize()), numNodes);
        VecList positions(this->opts.embeddingDimension, numNodes);
        std::vector<double> weights(numNodes);

#pragma omp parallel for default(none) shared(numNodes, positions, weights, indexToGraphMap) schedule(static)
        for (size_t i = 0; i < numNodes; i++) {
            positions[i] = this->currentPositions[indexToGraphMap[i]];
            weights[i] = this->currentWeights[indexToGraphMap[i]];
        }

        const std::vector<double> weightBuckets = WeightedIndex::getDoublingWeightBuckets(weights, this->opts.doublingFactor);
        currentWeightedIndex.updateIndices(positions, weights, weightBuckets);
    }
}

std::vector<NodeId> NewWEmbedEmbedder::getRepellingCandidatesForNode(NodeId v, VecBuffer<2> &buffer, WeightedIndex currentWeightedIndex, std::vector<NodeId>& indexToGraphMap) const {
    //TODO: Definitely think about refactoring this
    std::vector<NodeId> candidates;

    if (this->opts.numNegativeSamples >= 0) {
        candidates = sampleRandomNoise(std::min(static_cast<int32_t>(graphSize()), this->opts.numNegativeSamples));
        return candidates;
    }

    currentWeightedIndex.getNodesWithinWeightedDistance(this->currentPositions[v], this->currentWeights[v], this->opts.edgeLength,
                                                       candidates, buffer);
    for (NodeId& candidate: candidates) {
        candidate = indexToGraphMap[candidate];
        ASSERT(candidate < graphSize() && candidate >= 0, "Index out of bounds: " << candidate << " for N = " << graphSize());
    }
    return candidates;
}

void NewWEmbedEmbedder::calculateAllAttractingForces(VecList& force, std::vector<double>& weightParameterForce) {
    VecBuffer<1> buffer(this->opts.embeddingDimension);
    for (const NodeId v : this->sortedNodeIDs) {
        for (const NodeId u : graph.getNeighbors(v)) {
            attractionForce(v, u, force, buffer);
            attractionWeightForce(v, u, weightParameterForce, buffer);
        }
    }
}

void NewWEmbedEmbedder::calculateAllRepellingForces(WeightedIndex currentWeightedIndex, std::vector<NodeId>& indexToGraphMap) {
    VecBuffer<2> indexBuffer(this->opts.embeddingDimension);
    VecBuffer<1> forceBuffer(this->opts.embeddingDimension);
    numRepForceCalculations = 0;

#pragma omp parallel for firstprivate(indexBuffer, forceBuffer), reduction(+:numRepForceCalculations), schedule(runtime)

    for (const NodeId v : sortedNodeIDs) {
        std::vector<NodeId> repellingCandidates = getRepellingCandidatesForNode(v, indexBuffer, currentWeightedIndex, indexToGraphMap);
        for (const NodeId u : repellingCandidates) {
            if (graph.areNeighbors(v, u) || graph.areInSameColorClass(v, u)) {
                continue;
            }
            repellingForce(v, u, forceBuffer);
            repellingWeightForce(v, u, forceBuffer);
            numRepForceCalculations++;
        }
    }
}

//TODO: This could be moved somewhere else
std::vector<NodeId> NewWEmbedEmbedder::sampleRandomNoise(const int32_t numNodes) const {
    std::vector<NodeId> result;
    return Rand::randomSample(static_cast<int32_t>(graphSize()), numNodes);
}