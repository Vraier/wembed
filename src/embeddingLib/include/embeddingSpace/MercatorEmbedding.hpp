#pragma once

#include "Embedding.hpp"
#include "VecList.hpp"

class MercatorEmbedding : public Embedding {
   public:
    MercatorEmbedding(const std::vector<float>& radii, const std::vector<std::vector<float>>& positions);
    MercatorEmbedding(const std::vector<float>& radii, const std::vector<float>& thetas);
    virtual float getSimilarity(NodeId a, NodeId b) const;
    virtual int getDimension() const;

   private:
    const int DIMENSION;
    VecList coordinates;
    std::vector<float> thetas;
    std::vector<float> radii;

    float S1_distance(float r1, float r2, float theta1, float theta2) const;
    float compute_angle_d_vectors(CVecRef v1, CVecRef v2) const;
    float SD_distance(float r1, float r2, CVecRef pos1, CVecRef pos2) const;
};
