#include "Poincare.hpp"

Poincare::Poincare(const std::vector<std::vector<double>> &coords)
    : DIMENSION(coords[0].size()), coordinates(DIMENSION), buffer(DIMENSION) {
    coordinates.setSize(coords.size(), 0);

    for (int i = 0; i < coords.size(); i++) {
        ASSERT(coords[i].size() == DIMENSION,
               "Coord at position " << i << " has wrong dimension " << coords[i].size() << " instead of " << DIMENSION);
        for (int j = 0; j < DIMENSION; j++) {
            coordinates[i][j] = coords[i][j];
        }
    }
}

double Poincare::getSimilarity(NodeId a, NodeId b) {
    TmpVec<0> tmpVec(buffer);
    tmpVec = coordinates[a] - coordinates[b];
    double eps = 1e-5;

    // Squared norms, clamped
    double sqanorm = std::min(std::max(coordinates[a].sqNorm(), 0.0), 1.0 - eps);
    double sqbnorm = std::min(std::max(coordinates[b].sqNorm(), 0.0), 1.0 - eps);
    double sqdist = tmpVec.sqNorm();

    double x = (sqdist / ((1 - sqanorm) * (1 - sqbnorm))) * 2 + 1;
    double z = std::sqrt(std::pow(x, 2) - 1);
    return std::log(x + z);
}

int Poincare::getDimension() { return DIMENSION; }
