#include "Cosine.hpp"

Cosine::Cosine(const std::vector<std::vector<double>> &coords) : DIMENSION(coords[0].size()), coordinates(DIMENSION) {
    coordinates.setSize(coords.size(), 0);

    for (int i = 0; i < coords.size(); i++) {
        ASSERT(coords[i].size() == DIMENSION);
        for (int j = 0; j < DIMENSION; j++) {
            coordinates[i][j] = coords[i][j];
        }
    }
}

double Cosine::getSimilarity(NodeId a, NodeId b) {
    double aDotb = 0.0;
    double aNorm = coordinates[a].norm();
    double bNorm = coordinates[b].norm();
    for (int d = 0; d < DIMENSION; d++) {
        aDotb += coordinates[a][d] * coordinates[b][d];
    }

    // high values mean high edge probability.
    // my code assumes the opposite. Therefore i invert the value
    return -1.0 * aDotb / (aNorm * bNorm);
}

int Cosine::getDimension() {
    return DIMENSION;
}
