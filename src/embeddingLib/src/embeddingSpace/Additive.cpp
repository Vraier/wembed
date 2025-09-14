#include "Additive.hpp"
#include "VectorOperations.hpp"

Additive::Additive(const std::vector<std::vector<double>> &coords, const std::vector<double> &w)
    : DIMENSION(coords[0].size()), coordinates(DIMENSION), weights(w) {
    ASSERT(coords.size() == weights.size());

    coordinates.setSize(coords.size(), 0);
    for (int i = 0; i < coords.size(); i++) {
        ASSERT(coords[i].size() == DIMENSION);
        for (int j = 0; j < DIMENSION; j++) {
            coordinates[i][j] = coords[i][j];
        }
    }
}

double Additive::getSimilarity(NodeId a, NodeId b) const {
    VecBuffer<1> buffer(DIMENSION); // i allocate the buffer locally to avoid race conditions
    double dist = vectorOperations::calculateLPNorm(coordinates[a], coordinates[b]);
    return dist / (Toolkit::myPow(weights[a], 1.0 / DIMENSION) + Toolkit::myPow(weights[b], 1.0 / DIMENSION));
}

int Additive::getDimension() const { return DIMENSION; }

double Additive::getDistance(NodeId a, NodeId b) const {
    VecBuffer<1> buffer(DIMENSION);
    double dist = vectorOperations::calculateLPNorm(coordinates[a], coordinates[b]);
    return dist;
}

double Additive::getNodeWeight(NodeId a) const { return weights[a]; }
