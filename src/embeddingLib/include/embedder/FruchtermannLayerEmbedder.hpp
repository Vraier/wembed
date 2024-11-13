#pragma once

#include "AbstractLayerEmbedder.hpp"

class FruchtermannLayerEmbedder : public AbstractSimpleEmbedder {
   public:
    FruchtermannLayerEmbedder(OptionValues opts, GraphHierarchy *h, int level) : AbstractSimpleEmbedder(opts, h, level){};

   protected:
    /**
     * calculates all the attracting forces acting on this node.
     * also scales them accordingly
     */
    virtual TmpCVec<ATTR_BUFFER> attractionForce(int v, int level, EdgeId target);
    /**
     * calculates all the repelling forces acting on this node.
     * also scales them accordingly
     */
    virtual TmpCVec<REP_BUFFER> repulsionForce(int v, int levelV, int u, int levelU);
};
