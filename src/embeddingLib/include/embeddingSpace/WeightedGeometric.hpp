#pragma once

#include "Embedding.hpp"
#include "VecList.hpp"

class WeightedGeometric : public Embedding {
   public:
    WeightedGeometric(const std::vector<std::vector<float>> &coords, const std::vector<float> &weights, int p);
    virtual ~WeightedGeometric(){};

    virtual float getSimilarity(NodeId a, NodeId b) const;
    virtual int getDimension() const;
    float getDistance(NodeId a, NodeId b) const;
    float getNodeWeight(NodeId a) const;

   private:
    const int DIMENSION;
    const float DINVERSE;
    const int P;
    VecList coordinates;
    std::vector<float> weights;
};
