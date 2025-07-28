#include "WeightedGeometric.hpp"
#include "VectorOperations.hpp"

WeightedGeometric::WeightedGeometric(const std::vector<std::vector<double>> &coords, const std::vector<double> &w, int p)
    : DIMENSION(coords[0].size()), DINVERSE(1.0 / (double)DIMENSION), coordinates(DIMENSION), weights(w), P(p) {
    ASSERT(coords.size() == weights.size());

    coordinates.setSize(coords.size(), 0);
    for (int i = 0; i < coords.size(); i++) {
        ASSERT(coords[i].size() == DIMENSION);
        for (int j = 0; j < DIMENSION; j++) {
            coordinates[i][j] = coords[i][j];
        }
    }
}

double WeightedGeometric::getSimilarity(NodeId a, NodeId b) const {
    VecBuffer<1> buffer(DIMENSION); // i allocate the buffer locally to avoid race conditions
    double dist = vectorOperations::calculateLPNorm(coordinates[a], coordinates[b]);
    return dist / std::pow((weights[a] * weights[b]), DINVERSE);
}

int WeightedGeometric::getDimension() const { return DIMENSION; }

double WeightedGeometric::getDistance(NodeId a, NodeId b) const {
    VecBuffer<1> buffer(DIMENSION);
    double dist = vectorOperations::calculateLPNorm(coordinates[a], coordinates[b]);
    return dist;
}

double WeightedGeometric::getNodeWeight(NodeId a) const { return weights[a]; }
