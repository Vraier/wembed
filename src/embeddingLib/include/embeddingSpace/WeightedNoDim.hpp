#pragma once

#include "Embedding.hpp"
#include "VecList.hpp"


/**
 * Same as weighted geometric embedding (girg) but does not care about the dimension in the exponent
*/
class WeightedNoDim : public Embedding {
   public:
    WeightedNoDim(const std::vector<std::vector<float>> &coords, const std::vector<float> &weights);
    virtual ~WeightedNoDim(){};

    virtual float getSimilarity(NodeId a, NodeId b) const;
    virtual int getDimension() const;
    float getDistance(NodeId a, NodeId b) const;
    float getNodeWeight(NodeId a) const;

   private:
    const int DIMENSION;
    VecList coordinates;
    std::vector<float> weights;
};
