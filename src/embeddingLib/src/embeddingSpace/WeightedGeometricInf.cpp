#include "WeightedGeometricInf.hpp"

WeightedGeometricInf::WeightedGeometricInf(const std::vector<std::vector<double>> &coords,
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

double WeightedGeometricInf::getSimilarity(NodeId a, NodeId b) {
    TmpVec<0> tmpVec(buffer);
    tmpVec = coordinates[a] - coordinates[b];
    return tmpVec.infNorm() / std::pow((weights[a] * weights[b]), DINVERSE);
}

int WeightedGeometricInf::getDimension() {
    return DIMENSION;
}

double WeightedGeometricInf::getDistance(NodeId a, NodeId b) {
    TmpVec<0> tmpVec(buffer);
    tmpVec = coordinates[a] - coordinates[b];
    return tmpVec.infNorm();
}

double WeightedGeometricInf::getNodeWeight(NodeId a) const {
    return weights[a];
}
