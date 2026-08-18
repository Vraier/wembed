#pragma once

#include "Embedding.hpp"
#include "VecList.hpp"

/**
 * Two nodes are connected if |p_u-p_v| <= r_u^1/d + r_v^1/d
*/
class Additive : public Embedding {
   public:
    Additive(const std::vector<std::vector<float>> &coords, const std::vector<float> &weights);
    virtual ~Additive(){};

    virtual float getSimilarity(NodeId a, NodeId b) const;
    virtual int getDimension() const;
    float getDistance(NodeId a, NodeId b) const;
    float getNodeWeight(NodeId a) const;

   private:
    const int DIMENSION;
    VecList coordinates;
    std::vector<float> weights;
};
