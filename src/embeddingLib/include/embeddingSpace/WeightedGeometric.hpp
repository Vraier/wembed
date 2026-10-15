#pragma once

#include "Embedding.hpp"
#include "VecList.hpp"

class WeightedGeometric : public Embedding {
   public:
    WeightedGeometric(const std::vector<std::vector<flt_t>> &coords, const std::vector<flt_t> &weights, int p);
    virtual ~WeightedGeometric(){};

    virtual flt_t getSimilarity(NodeId a, NodeId b) const;
    virtual int getDimension() const;
    flt_t getDistance(NodeId a, NodeId b) const;
    flt_t getNodeWeight(NodeId a) const;

   private:
    const int DIMENSION;
    const flt_t DINVERSE;
    const int P;
    VecList<> coordinates;
    std::vector<flt_t> weights;
};
