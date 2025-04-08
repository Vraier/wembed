#pragma once

#include "Embedding.hpp"
#include "VecList.hpp"

class MercatorEmbedding : public Embedding {
   public:
    MercatorEmbedding(const std::vector<double>& radii, const std::vector<std::vector<double>>& positions);
    MercatorEmbedding(const std::vector<double>& radii, const std::vector<double>& thetas);
    virtual double getSimilarity(NodeId a, NodeId b) const;
    virtual int getDimension() const;

   private:
    const int DIMENSION;
    VecList coordinates;
    std::vector<double> thetas;
    std::vector<double> radii;

    double S1_distance(double r1, double r2, double theta1, double theta2) const;
    double compute_angle_d_vectors(CVecRef v1, CVecRef v2) const;
    double SD_distance(double r1, double r2, CVecRef pos1, CVecRef pos2) const;
};
