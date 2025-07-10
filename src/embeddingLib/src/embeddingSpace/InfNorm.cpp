#include "InfNorm.hpp"

InfNorm::InfNorm(const std::vector<std::vector<double>> &coords)
    : DIMENSION(coords[0].size()), coordinates(DIMENSION) {
    coordinates.setSize(coords.size(), 0);

    for (int i = 0; i < coords.size(); i++) {
        ASSERT(coords[i].size() == DIMENSION, "Coordinate at index " + std::to_string(i) + " has dimension " +
                                                  std::to_string(coords[i].size()) + " but expected " +
                                                  std::to_string(DIMENSION));
        for (int j = 0; j < DIMENSION; j++) {
            coordinates[i][j] = coords[i][j];
        }
    }
}

double InfNorm::getSimilarity(NodeId a, NodeId b) const {
    VecBuffer<1> buffer(DIMENSION);
    TmpVec<0> tmpVec(buffer);
    tmpVec = coordinates[a] - coordinates[b];
    return tmpVec.infNorm();
}

int InfNorm::getDimension() const { return DIMENSION; }
