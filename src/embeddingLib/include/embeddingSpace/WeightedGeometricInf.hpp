#pragma once

#include "Embedding.hpp"
#include "VecList.hpp"

class WeightedGeometricInf : public Embedding {
   public:
    WeightedGeometricInf(const std::vector<std::vector<flt_t>> &coords, const std::vector<flt_t> &weights);
    virtual ~WeightedGeometricInf(){};

    virtual flt_t getSimilarity(NodeId a, NodeId b) const;
    virtual int getDimension() const;
    flt_t getDistance(NodeId a, NodeId b) const;
    flt_t getNodeWeight(NodeId a) const;

   private:
    const int DIMENSION;
    const flt_t DINVERSE;
    VecList<> coordinates;
    std::vector<flt_t> weights;
};
