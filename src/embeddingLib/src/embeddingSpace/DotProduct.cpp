#include "DotProduct.hpp"

DotProduct::DotProduct(const std::vector<std::vector<double>> &coords)
    : DIMENSION(coords[0].size()), coordinates(DIMENSION) {
    coordinates.setSize(coords.size(), 0);

    for (int i = 0; i < coords.size(); i++) {
        ASSERT(coords[i].size() == DIMENSION,
               "Coord at position " << i << " has wrong dimension " << coords[i].size() << " instead of " << DIMENSION);
        for (int j = 0; j < DIMENSION; j++) {
            coordinates[i][j] = coords[i][j];
        }
    }
}

double DotProduct::getSimilarity(NodeId a, NodeId b) {
    double res = 0.0;
    for (int d = 0; d < DIMENSION; d++) {
        res += coordinates[a][d] * coordinates[b][d];
    }
    // high values mean high edge probability.
    // my code assumes the opposite. Therefore i invert the value
    res *= -1;
    return res;
}

int DotProduct::getDimension() { return DIMENSION; }
