#include "WeightedGeometricInf.hpp"

WeightedGeometricInf::WeightedGeometricInf(const std::vector<std::vector<flt_t>> &coords,
                                     const std::vector<flt_t> &w) : DIMENSION(coords[0].size()),
                                                                     DINVERSE(1.0 / (flt_t)DIMENSION),
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

flt_t WeightedGeometricInf::getSimilarity(NodeId a, NodeId b) const {
    VecBuffer<1> buffer(DIMENSION);
    TmpVec<0> tmpVec(buffer);
    tmpVec = coordinates[a] - coordinates[b];
    return tmpVec.infNorm() / std::pow((weights[a] * weights[b]), DINVERSE);
}

int WeightedGeometricInf::getDimension() const {
    return DIMENSION;
}

flt_t WeightedGeometricInf::getDistance(NodeId a, NodeId b) const {
    VecBuffer<1> buffer(DIMENSION);
    TmpVec<0> tmpVec(buffer);
    tmpVec = coordinates[a] - coordinates[b];
    return tmpVec.infNorm();
}

flt_t WeightedGeometricInf::getNodeWeight(NodeId a) const {
    return weights[a];
}
