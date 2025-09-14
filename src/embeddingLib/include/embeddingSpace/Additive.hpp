#pragma once

#include "Embedding.hpp"
#include "VecList.hpp"

/**
 * Two nodes are connected if |p_u-p_v| <= r_u^1/d + r_v^1/d
*/
class Additive : public Embedding {
   public:
    Additive(const std::vector<std::vector<double>> &coords, const std::vector<double> &weights);
    virtual ~Additive(){};

    virtual double getSimilarity(NodeId a, NodeId b) const;
    virtual int getDimension() const;
    double getDistance(NodeId a, NodeId b) const;
    double getNodeWeight(NodeId a) const;

   private:
    const int DIMENSION;
    VecList coordinates;
    std::vector<double> weights;
};
