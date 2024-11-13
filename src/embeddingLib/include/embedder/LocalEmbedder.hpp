#pragma once

#include "EmbedderOptions.hpp"
#include "EmbeddedGraph.hpp"
#include "Graph.hpp"
#include "VecList.hpp"
#include "AbstractSimpleEmbedder.hpp"

/**
 * Uses Thomas proposal of moving vertices with sigmoid function according to furthest neighbor
 */
class LocalEmbedder : public AbstractSimpleEmbedder {
   public:
    LocalEmbedder(Graph& g, EmbedderOptions opts) : AbstractSimpleEmbedder(g, opts), maxNeighborDist(g.getNumVertices()) {};

    
   protected:
    virtual void calculateStep() override;

    virtual TmpCVec<REP_BUFFER> repulsionForce(int v, int u);
    virtual TmpCVec<ATTR_BUFFER> attractionForce(int v, int u);

    double idealEdgeLength(double wa, double wb);
    double sigmoid(double x); // sigmoid function
    double invSigmoid(double x); // decreasing sigmoid function
    void updateFurthestDistances();


    // for node u holds the distance of the neighbor that is the furthest away from u
    std::vector<double> maxNeighborDist; 
};
