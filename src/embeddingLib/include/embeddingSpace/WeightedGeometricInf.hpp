#pragma once

#include "Embedding.hpp"
#include "VecList.hpp"

class WeightedGeometricInf : public Embedding {
   public:
    WeightedGeometricInf(const std::vector<std::vector<float>> &coords, const std::vector<float> &weights);
    virtual ~WeightedGeometricInf(){};

    virtual float getSimilarity(NodeId a, NodeId b) const;
    virtual int getDimension() const;
    float getDistance(NodeId a, NodeId b) const;
    float getNodeWeight(NodeId a) const;

   private:
    const int DIMENSION;
    const float DINVERSE;
    VecList coordinates;
    std::vector<float> weights;
};
