#include "MercatorEmbedding.hpp"

#include <iostream>

MercatorEmbedding::MercatorEmbedding(const std::vector<float>& radii,
                                     const std::vector<std::vector<float>>& positions)
    : DIMENSION(positions[0].size() - 1), coordinates(positions[0].size()) {
    ASSERT(radii.size() == positions.size());
    ASSERT(DIMENSION > 0);
    this->radii = radii;
    coordinates.setSize(positions.size(), 0);
    for (int i = 0; i < positions.size(); i++) {
        ASSERT(positions[i].size() == positions[0].size());
        for (int j = 0; j < positions[0].size(); j++) {
            coordinates[i][j] = positions[i][j];
        }
    }
}

MercatorEmbedding::MercatorEmbedding(const std::vector<float>& radii, const std::vector<float>& thetas)
    : DIMENSION(1), coordinates(DIMENSION) {
    ASSERT(radii.size() == thetas.size());
    this->radii = radii;
    this->thetas = thetas;
}

float MercatorEmbedding::getSimilarity(NodeId a, NodeId b) const {
    if (DIMENSION == 1)
        return S1_distance(radii[a], radii[b], thetas[a], thetas[b]);
    else
        return SD_distance(radii[a], radii[b], coordinates[a], coordinates[b]);
}

int MercatorEmbedding::getDimension() const { return DIMENSION; }

// https://github.com/networkgeometry/d-mercator/blob/b259bd0194ad7394f76bef3de681273f479c881d/lib/greedy_routing.cpp#L170
float MercatorEmbedding::S1_distance(float r1, float r2, float theta1, float theta2) const {
    if ((r1 == r2) && (theta1 == theta2)) {
        return 0;
    }
    float delta_theta = M_PI - std::fabs(M_PI - std::fabs(theta1 - theta2));
    if (delta_theta == 0) {
        return std::fabs(r1 - r2);
    } else {
        auto dist =
            0.5 * ((1 - std::cos(delta_theta)) * std::cosh(r1 + r2) + (1 + std::cos(delta_theta)) * std::cosh(r1 - r2));
        return std::acosh(dist);
    }
}

float MercatorEmbedding::compute_angle_d_vectors(CVecRef v1, CVecRef v2) const {
    ASSERT(v1.dimension() == v2.dimension());
    float angle{0}, norm1{0}, norm2{0};
    for (int i = 0; i < v1.dimension(); ++i) {
        angle += v1[i] * v2[i];
        norm1 += v1[i] * v1[i];
        norm2 += v2[i] * v2[i];
    }
    norm1 /= sqrt(norm1);
    norm2 /= sqrt(norm2);

    const auto result = angle / (norm1 * norm2);
    if (std::fabs(result - 1) < 1e-15)
        return 0;
    else
        return std::acos(result);
}

float MercatorEmbedding::SD_distance(float r1, float r2, CVecRef pos1, CVecRef pos2) const {
    float delta_theta = compute_angle_d_vectors(pos1, pos2);
    if ((r1 == r2) && delta_theta == 0) {
        return 0;  // the same positions
    }

    if (delta_theta == 0) {
        return std::fabs(r1 - r2);
    } else {
        auto dist =
            0.5 * ((1 - std::cos(delta_theta)) * std::cosh(r1 + r2) + (1 + std::cos(delta_theta)) * std::cosh(r1 - r2));
        return std::acosh(dist);
    }
}
