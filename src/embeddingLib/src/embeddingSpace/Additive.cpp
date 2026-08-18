#include "Additive.hpp"
#include "VectorOperations.hpp"

Additive::Additive(const std::vector<std::vector<float>> &coords, const std::vector<float> &w)
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

float Additive::getSimilarity(NodeId a, NodeId b) const {
    VecBuffer<1> buffer(DIMENSION); // i allocate the buffer locally to avoid race conditions
    float dist = vectorOperations::calculateLPNormf(coordinates[a], coordinates[b]);
    return dist / (Toolkit::myPowf(weights[a], 1.0f / DIMENSION) + Toolkit::myPowf(weights[b], 1.0f / DIMENSION));
}

int Additive::getDimension() const { return DIMENSION; }

float Additive::getDistance(NodeId a, NodeId b) const {
    VecBuffer<1> buffer(DIMENSION);
    float dist = vectorOperations::calculateLPNormf(coordinates[a], coordinates[b]);
    return dist;
}

float Additive::getNodeWeight(NodeId a) const { return weights[a]; }
