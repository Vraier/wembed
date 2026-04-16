#include "NewWEmbedEmbedder.hpp"

void NewWEmbedEmbedder::calculateStep() {
    //TODO: Advance embedding by single gradient descent
}

bool NewWEmbedEmbedder::isFinished() {
    //TODO: Add check for insignificant change
    return this->currentIteration >= this->opts.maxIterations;
}

void NewWEmbedEmbedder::calculateEmbedding() {
    /*TODO: Start Timer
     * repeatedly call calculateStep
     * Stop Timer
     */
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
    //TODO: Verify coordinates and set this->currentCoordinates

}

void NewWEmbedEmbedder::setWeights(const std::vector<double> &weights) {
    /*TODO: Set Weights
     * update member variables
     */
}
