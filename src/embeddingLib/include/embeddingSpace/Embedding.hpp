#pragma once

#include "Graph.hpp"

class Embedding {
   public:
    virtual ~Embedding(){};

    /**
     * Returns the similarity of the two given nodes. 
     * CAUTION: A low values indicates a high similarity. A high value indicates a low similarity.
     * Think of it like the euclidean distance.
    */
    virtual double getSimilarity(NodeId a, NodeId b) = 0;
    virtual int getDimension() = 0;
};
