#include "FruchtermannLayerEmbedder.hpp"

TmpCVec<0> FruchtermannLayerEmbedder::attractionForce(int v, int level, EdgeId target) {
    ASSERT(LEVEL == level);
    TmpVec<0> result(buffer, 0.0);

    NodeId u = graph.getEdgeTarget(target);
    result = hierarchy->getAveragePosition(LEVEL, u) - hierarchy->getAveragePosition(LEVEL, v);
    double norm = result.norm();

    if (norm <= 0) {
        LOG_WARNING("Random displacement attr V: (" << v << "," << LEVEL << ") U: (" << u << "," << LEVEL << ")");
        result.setToRandomUnitVector();
        return result;
    }

    double edgeWeight = hierarchy->getInverseEdgeWeightSum(LEVEL, target);
    double factor = 1.0;

    factor *= std::pow(norm, options.embedderOptions.forceExponent);
    factor /= (double)hierarchy->getTotalContainedNodes(LEVEL, v);
    factor *= (double)edgeWeight / options.embedderOptions.cSpring;
    factor /= norm;

    result *= factor;
    return result;
}

TmpCVec<1> FruchtermannLayerEmbedder::repulsionForce(int v, int levelV, int u, int levelU) {
    ASSERT(levelV == LEVEL);
    TmpVec<1> result(buffer, 0.0);

    if (v == u && levelV == levelU) return result;

    result = hierarchy->getAveragePosition(levelV, v) - hierarchy->getAveragePosition(levelU, u);
    double norm = result.norm();

    if (norm <= 0) {
        LOG_WARNING("Random displacement rep V: (" << v << "," << levelV << ") U: (" << u << "," << levelU << ")");
        result.setToRandomUnitVector();
        return result;
    }

    double weightV = hierarchy->getScaledWeightSum(levelV, v);
    double weightU = hierarchy->getScaledWeightSum(levelU, u);

    double factor = 1.0;
    factor *= (weightV * weightU);
    factor *= options.embedderOptions.cSpring;
    factor /= norm * norm;
    factor /= (double)hierarchy->getTotalContainedNodes(levelV, v);

    result *= factor;
    return result;
}
