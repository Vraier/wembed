#pragma once

#include "AbstractSimpleEmbedder.hpp"
/**
 * Calculates a weighted embedding using a modified Fruchterman approach 
 * Calculates all pair repulsion forces -> O(n^2)
*/
class FruchtermannWeightEmbedder: public AbstractSimpleEmbedder {
   public:
    FruchtermannWeightEmbedder(Graph& g, OptionValues opts) : AbstractSimpleEmbedder(g, opts){};

   protected:

    virtual TmpCVec<REP_BUFFER> repulsionForce(int v, int u);
    virtual TmpCVec<ATTR_BUFFER> attractionForce(int u, int v);

    double idealEdgeLength(double wa, double wb);

    static constexpr int WEIGHT_BUFFER = 5;

    virtual void calculateWeightStep();
    virtual double getOptimalWeight(NodeId a);
    virtual double getOptimalWeightUpperBound(NodeId a);
    virtual double getOptimalWeightWeightedError(NodeId a);
    virtual double getOptimalWeightExactError(NodeId a);
};
