#include "NewWEmbedEmbedder.hpp"

#include <fstream>


// ======================================================================================
//
//                       PUBLIC FUNCTIONS NewWEmbedEmbedder
//
// ======================================================================================
void NewWEmbedEmbedder::calculateStep() {
    /* TODO:
     * Update indices
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
    VecList oldPositions(this->currentPositions.dimension(), this->currentPositions.size());
    std::vector<double> weightParameterForce(graphSize(), 0);



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
