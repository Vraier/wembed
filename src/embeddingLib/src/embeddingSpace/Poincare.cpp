#include "Poincare.hpp"

Poincare::Poincare(const std::vector<std::vector<flt_t>> &coords)
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

flt_t Poincare::getSimilarity(NodeId a, NodeId b) const {
    VecBuffer<1> buffer(DIMENSION);
    TmpVec<0> tmpVec(buffer);
    tmpVec = coordinates[a] - coordinates[b];
    flt_t eps = 1e-5;

    // Squared norms, clamped
    flt_t sqanorm = std::min(std::max(coordinates[a].sqNorm(), flt_t{0.0}), flt_t{1.0} - eps);
    flt_t sqbnorm = std::min(std::max(coordinates[b].sqNorm(), flt_t{0.0}), flt_t{1.0} - eps);
    flt_t sqdist = tmpVec.sqNorm();

    flt_t x = (sqdist / ((1 - sqanorm) * (1 - sqbnorm))) * 2 + 1;
    flt_t z = std::sqrt(std::pow(x, 2.f) - 1.f);
    return std::log(x + z);
}

int Poincare::getDimension() const { return DIMENSION; }
