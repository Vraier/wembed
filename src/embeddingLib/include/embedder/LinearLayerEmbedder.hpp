#pragma once

#include "AbstractLayerEmbedder.hpp"

/**
 * Emulates sigmoid loss with a linear function.
 * Does incorporate desired edge length and sigmoid scale (as well as sigmoid slack).
 */
class LinearLayerEmbedder : public AbstractSimpleEmbedder {
   public:
    LinearLayerEmbedder(OptionValues opts, GraphHierarchy *h, int level) : AbstractSimpleEmbedder(opts, h, level){};

   protected:
    /**
     * calculates all the attracting forces acting on this node.
     */
    virtual TmpCVec<ATTR_BUFFER> attractionForce(int v, int level, EdgeId target);
    /**
     * calculates all the repelling forces acting on this node.
     */
    virtual TmpCVec<REP_BUFFER> repulsionForce(int v, int levelV, int u, int levelU);

    double idealEdgeLength(double weightV, double weightU);
};
