#include "WeightedNoDim.hpp"

WeightedNoDim::WeightedNoDim(const std::vector<std::vector<double>> &coords,
                                     const std::vector<double> &w) : DIMENSION(coords[0].size()),
                                                                     coordinates(DIMENSION),
                                                                     weights(w),
                                                                     buffer(DIMENSION) {
    ASSERT(coords.size() == weights.size());

    coordinates.setSize(coords.size(), 0);
    for (int i = 0; i < coords.size(); i++) {
        ASSERT(coords[i].size() == DIMENSION);
        for (int j = 0; j < DIMENSION; j++) {
            coordinates[i][j] = coords[i][j];
        }
    }
}

double WeightedNoDim::getSimilarity(NodeId a, NodeId b) {
    TmpVec<0> tmpVec(buffer);
    tmpVec = coordinates[a] - coordinates[b];
    return tmpVec.norm() / (weights[a] * weights[b]);
}

int WeightedNoDim::getDimension() {
    return DIMENSION;
}

double WeightedNoDim::getDistance(NodeId a, NodeId b) {
    TmpVec<0> tmpVec(buffer);
    tmpVec = coordinates[a] - coordinates[b];
    return tmpVec.norm();
}

double WeightedNoDim::getNodeWeight(NodeId a) const {
    return weights[a];
}
