#include "WeightedGeometric.hpp"
#include "VectorOperations.hpp"

WeightedGeometric::WeightedGeometric(const std::vector<std::vector<float>> &coords, const std::vector<float> &w, int p)
    : DIMENSION(coords[0].size()), DINVERSE(1.0 / (float)DIMENSION), coordinates(DIMENSION), weights(w), P(p) {
    ASSERT(coords.size() == weights.size());

    coordinates.setSize(coords.size(), 0);
    for (int i = 0; i < coords.size(); i++) {
        ASSERT(coords[i].size() == DIMENSION);
        for (int j = 0; j < DIMENSION; j++) {
            coordinates[i][j] = coords[i][j];
        }
    }
}

float WeightedGeometric::getSimilarity(NodeId a, NodeId b) const {
    VecBuffer<1> buffer(DIMENSION); // i allocate the buffer locally to avoid race conditions
    float dist = vectorOperations::calculateLPNormf(coordinates[a], coordinates[b]);
    return dist / std::pow((weights[a] * weights[b]), DINVERSE);
}

int WeightedGeometric::getDimension() const { return DIMENSION; }

float WeightedGeometric::getDistance(NodeId a, NodeId b) const {
    VecBuffer<1> buffer(DIMENSION);
    float dist = vectorOperations::calculateLPNormf(coordinates[a], coordinates[b]);
    return dist;
}

float WeightedGeometric::getNodeWeight(NodeId a) const { return weights[a]; }
