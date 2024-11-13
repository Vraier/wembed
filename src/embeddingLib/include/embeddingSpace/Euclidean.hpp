#pragma once

#include "Embedding.hpp"
#include "VecList.hpp"

class Euclidean : public Embedding {
   public:
    Euclidean(const std::vector<std::vector<double>> &coords);
    virtual double getSimilarity(NodeId a, NodeId b);
    virtual int getDimension();

   private:
    const int DIMENSION;
    VecList coordinates;
    VecBuffer<1> buffer;
};
