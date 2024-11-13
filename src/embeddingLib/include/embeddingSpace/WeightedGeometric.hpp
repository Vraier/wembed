#pragma once

#include "Embedding.hpp"
#include "VecList.hpp"

class WeightedGeometric : public Embedding {
   public:
    WeightedGeometric(const std::vector<std::vector<double>> &coords, const std::vector<double> &weights);
    virtual ~WeightedGeometric(){};

    virtual double getSimilarity(NodeId a, NodeId b);
    virtual int getDimension();
    double getDistance(NodeId a, NodeId b);
    double getNodeWeight(NodeId a) const;

   private:
    const int DIMENSION;
    const double DINVERSE;
    VecList coordinates;
    std::vector<double> weights;
    VecBuffer<1> buffer;
};
