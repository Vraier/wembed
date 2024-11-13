#pragma once

#include "Embedding.hpp"
#include "VecList.hpp"


/**
 * Same as weighted geometric embedding (girg) but does not care about the dimension in the exponent
*/
class WeightedNoDim : public Embedding {
   public:
    WeightedNoDim(const std::vector<std::vector<double>> &coords, const std::vector<double> &weights);
    virtual ~WeightedNoDim(){};

    virtual double getSimilarity(NodeId a, NodeId b);
    virtual int getDimension();
    double getDistance(NodeId a, NodeId b);
    double getNodeWeight(NodeId a) const;

   private:
    const int DIMENSION;
    VecList coordinates;
    std::vector<double> weights;
    VecBuffer<1> buffer;
};
