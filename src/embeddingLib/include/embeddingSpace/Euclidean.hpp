#pragma once

#include "Embedding.hpp"
#include "VecList.hpp"

class Euclidean : public Embedding {
   public:
    Euclidean(const std::vector<std::vector<float>> &coords);
    virtual float getSimilarity(NodeId a, NodeId b) const;
    virtual int getDimension() const;

   private:
    const int DIMENSION;
    VecList coordinates;
};
