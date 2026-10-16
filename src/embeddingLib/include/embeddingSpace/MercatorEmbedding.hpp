#pragma once

#include "Embedding.hpp"
#include "VecList.hpp"

class MercatorEmbedding : public Embedding {
   public:
    MercatorEmbedding(const std::vector<flt_t>& radii, const std::vector<std::vector<flt_t>>& positions);
    MercatorEmbedding(const std::vector<flt_t>& radii, const std::vector<flt_t>& thetas);
    virtual flt_t getSimilarity(NodeId a, NodeId b) const;
    virtual int getDimension() const;

   private:
    const int DIMENSION;
    VecList<> coordinates;
    std::vector<flt_t> thetas;
    std::vector<flt_t> radii;

    flt_t S1_distance(flt_t r1, flt_t r2, flt_t theta1, flt_t theta2) const;
    flt_t compute_angle_d_vectors(CVecRef v1, CVecRef v2) const;
    flt_t SD_distance(flt_t r1, flt_t r2, CVecRef pos1, CVecRef pos2) const;
};
