#include "WeightedGeometric.hpp"

WeightedGeometric::WeightedGeometric(const std::vector<std::vector<double>> &coords,
                                     const std::vector<double> &w) : DIMENSION(coords[0].size()),
                                                                     DINVERSE(1.0 / (double)DIMENSION),
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

double WeightedGeometric::getSimilarity(NodeId a, NodeId b) {
    TmpVec<0> tmpVec(buffer);
    tmpVec = coordinates[a] - coordinates[b];
    return tmpVec.norm() / std::pow((weights[a] * weights[b]), DINVERSE);
}

int WeightedGeometric::getDimension() {
    return DIMENSION;
}

double WeightedGeometric::getDistance(NodeId a, NodeId b) {
    TmpVec<0> tmpVec(buffer);
    tmpVec = coordinates[a] - coordinates[b];
    return tmpVec.norm();
}

double WeightedGeometric::getNodeWeight(NodeId a) const {
    return weights[a];
}
