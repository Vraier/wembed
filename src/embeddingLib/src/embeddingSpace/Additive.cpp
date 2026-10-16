#include "Additive.hpp"
#include "VectorOperations.hpp"

Additive::Additive(const std::vector<std::vector<flt_t>> &coords, const std::vector<flt_t> &w)
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

flt_t Additive::getSimilarity(NodeId a, NodeId b) const {
    VecBuffer<1> buffer(DIMENSION); // i allocate the buffer locally to avoid race conditions
    flt_t dist = vectorOperations::calculateLPNorm(coordinates[a], coordinates[b]);
    return dist / (Toolkit::myPow(weights[a], flt_t{1.0} / static_cast<flt_t>(DIMENSION)) +
                   Toolkit::myPow(weights[b], flt_t{1.0} / static_cast<flt_t>(DIMENSION)));
}

int Additive::getDimension() const { return DIMENSION; }

flt_t Additive::getDistance(NodeId a, NodeId b) const {
    VecBuffer<1> buffer(DIMENSION);
    flt_t dist = vectorOperations::calculateLPNorm(coordinates[a], coordinates[b]);
    return dist;
}

flt_t Additive::getNodeWeight(NodeId a) const { return weights[a]; }
