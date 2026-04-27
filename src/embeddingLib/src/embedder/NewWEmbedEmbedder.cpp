#include "NewWEmbedEmbedder.hpp"

#include <fstream>

#include "WeightedIndex.hpp"


// ======================================================================================
//
//                       PUBLIC FUNCTIONS NewWEmbedEmbedder
//
// ======================================================================================
void NewWEmbedEmbedder::calculateStep() {
    /* TODO:
     * calculate attracting forces
     * calculate repelling forces
     * calculate weight Penalties
     * update positions
     * update weights
     * compute change in positions
     */

    //Increase current step
    this->currentIteration++;

    //TODO: This could only be compiled in debug mode
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

    this->timer->startTiming("attracting_forces", "Compute Attracting Forces");
    calculateAllAttractingForces();
    this->timer->stopTiming("attracting_forces");

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
    //TODO: Update hidden parameters
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

void NewWEmbedEmbedder::calculateAllAttractingForces() {
    //TODO:
}
