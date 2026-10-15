#pragma once

#include "Embedding.hpp"
#include "VecList.hpp"

/**
 * Two nodes are connected if |p_u-p_v| <= r_u^1/d + r_v^1/d
*/
class Additive : public Embedding {
   public:
    Additive(const std::vector<std::vector<flt_t>> &coords, const std::vector<flt_t> &weights);
    virtual ~Additive(){};

    virtual flt_t getSimilarity(NodeId a, NodeId b) const;
    virtual int getDimension() const;
    flt_t getDistance(NodeId a, NodeId b) const;
    flt_t getNodeWeight(NodeId a) const;

   private:
    const int DIMENSION;
    VecList<> coordinates;
    std::vector<flt_t> weights;
};
