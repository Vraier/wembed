#include "WeightedGeometric.hpp"
#include "VectorOperations.hpp"

WeightedGeometric::WeightedGeometric(const std::vector<std::vector<flt_t>> &coords, const std::vector<flt_t> &w, int p)
    : DIMENSION(coords[0].size()), DINVERSE(1.0 / (flt_t)DIMENSION), coordinates(DIMENSION), weights(w), P(p) {
    ASSERT(coords.size() == weights.size());

    coordinates.setSize(coords.size(), 0);
    for (int i = 0; i < coords.size(); i++) {
        ASSERT(coords[i].size() == DIMENSION);
        for (int j = 0; j < DIMENSION; j++) {
            coordinates[i][j] = coords[i][j];
        }
    }
}

flt_t WeightedGeometric::getSimilarity(NodeId a, NodeId b) const {
    VecBuffer<1> buffer(DIMENSION); // i allocate the buffer locally to avoid race conditions
    flt_t dist = vectorOperations::calculateLPNorm(coordinates[a], coordinates[b]);
    return dist / std::pow((weights[a] * weights[b]), DINVERSE);
}

int WeightedGeometric::getDimension() const { return DIMENSION; }

flt_t WeightedGeometric::getDistance(NodeId a, NodeId b) const {
    VecBuffer<1> buffer(DIMENSION);
    flt_t dist = vectorOperations::calculateLPNorm(coordinates[a], coordinates[b]);
    return dist;
}

flt_t WeightedGeometric::getNodeWeight(NodeId a) const { return weights[a]; }
