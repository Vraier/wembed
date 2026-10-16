#pragma once

#include "Embedding.hpp"
#include "VecList.hpp"


/**
 * Same as weighted geometric embedding (girg) but does not care about the dimension in the exponent
*/
class WeightedNoDim : public Embedding {
   public:
    WeightedNoDim(const std::vector<std::vector<flt_t>> &coords, const std::vector<flt_t> &weights);
    virtual ~WeightedNoDim(){};

    virtual flt_t getSimilarity(NodeId a, NodeId b) const;
    virtual int getDimension() const;
    flt_t getDistance(NodeId a, NodeId b) const;
    flt_t getNodeWeight(NodeId a) const;

   private:
    const int DIMENSION;
    VecList<> coordinates;
    std::vector<flt_t> weights;
};
