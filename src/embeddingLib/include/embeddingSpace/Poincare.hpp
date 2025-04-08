#pragma once

#include "Embedding.hpp"
#include "VecList.hpp"

class Poincare : public Embedding {
   public:
    Poincare(const std::vector<std::vector<double>> &coords);
    virtual double getSimilarity(NodeId a, NodeId b) const;
    virtual int getDimension() const;

   private:
    const int DIMENSION;
    VecList coordinates;
};
