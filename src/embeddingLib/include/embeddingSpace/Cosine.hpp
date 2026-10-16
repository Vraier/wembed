#pragma once

#include "Embedding.hpp"
#include "VecList.hpp"

class Cosine : public Embedding {
   public:
    Cosine(const std::vector<std::vector<flt_t>> &coords);
    virtual flt_t getSimilarity(NodeId a, NodeId b) const;
    virtual int getDimension() const;

   private:
    const int DIMENSION;
    VecList<> coordinates;
};
