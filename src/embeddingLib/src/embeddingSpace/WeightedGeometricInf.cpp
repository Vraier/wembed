#include "WeightedGeometricInf.hpp"

WeightedGeometricInf::WeightedGeometricInf(const std::vector<std::vector<float>> &coords,
                                     const std::vector<float> &w) : DIMENSION(coords[0].size()),
                                                                     DINVERSE(1.0 / (float)DIMENSION),
                                                                     coordinates(DIMENSION),
                                                                     weights(w){
    ASSERT(coords.size() == weights.size());

    coordinates.setSize(coords.size(), 0);
    for (int i = 0; i < coords.size(); i++) {
        ASSERT(coords[i].size() == DIMENSION);
        for (int j = 0; j < DIMENSION; j++) {
            coordinates[i][j] = coords[i][j];
        }
    }
}

float WeightedGeometricInf::getSimilarity(NodeId a, NodeId b) const {
    VecBuffer<1> buffer(DIMENSION);
    TmpVec<0> tmpVec(buffer);
    tmpVec = coordinates[a] - coordinates[b];
    return tmpVec.infNorm() / std::pow((weights[a] * weights[b]), DINVERSE);
}

int WeightedGeometricInf::getDimension() const {
    return DIMENSION;
}

float WeightedGeometricInf::getDistance(NodeId a, NodeId b) const {
    VecBuffer<1> buffer(DIMENSION);
    TmpVec<0> tmpVec(buffer);
    tmpVec = coordinates[a] - coordinates[b];
    return tmpVec.infNorm();
}

float WeightedGeometricInf::getNodeWeight(NodeId a) const {
    return weights[a];
}
