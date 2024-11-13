#pragma once

#include "Embedding.hpp"
#include "VecList.hpp"

class DotProduct : public Embedding {
   public:
    DotProduct(const std::vector<std::vector<double>> &coords);
    virtual double getSimilarity(NodeId a, NodeId b);
    virtual int getDimension();

   private:
    const int DIMENSION;
    VecList coordinates;
};
