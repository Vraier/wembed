#pragma once

#include "EmbOptions.hpp"
#include "EmbeddedGraph.hpp"
#include "Graph.hpp"
#include "VecList.hpp"
#include "AbstractSimpleEmbedder.hpp"

/**
 * Calculates a weighted embedding using a modified Fruchterman approach
 * Calculates all pair repulsion forces -> O(n^2)
 */
class SigmoidEmbedder : public AbstractSimpleEmbedder {
   public:
    SigmoidEmbedder(Graph& g, OptionValues opts) : AbstractSimpleEmbedder(g, opts) {};

    
   protected:
    virtual TmpCVec<REP_BUFFER> repulsionForce(int v, int u);
    virtual TmpCVec<ATTR_BUFFER> attractionForce(int v, int u);
    virtual double weightRepulsionForce(NodeId v, NodeId u);
    virtual double weightAttractionForce(NodeId v, NodeId u);

    double idealEdgeLength(double wa, double wb);
    double sigmoid(double x); // sigmoid function
    double invSigmoid(double x); // decreasing sigmoid function
};
