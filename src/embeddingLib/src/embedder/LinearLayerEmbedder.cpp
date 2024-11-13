#include "LinearLayerEmbedder.hpp"

TmpCVec<0> LinearLayerEmbedder::attractionForce(int v, int level, EdgeId target) {
    ASSERT(LEVEL == level);
    TmpVec<0> result(buffer, 0.0);

    NodeId u = graph.getEdgeTarget(target);

    if (v == u) return result;

    result = hierarchy->getAveragePosition(LEVEL, u) - hierarchy->getAveragePosition(LEVEL, v);
    double norm = result.norm();

    if (norm <= 0) {
        LOG_WARNING("Random displacement attr V: (" << v << "," << LEVEL << ") U: (" << u << "," << LEVEL << ")");
        result.setToRandomUnitVector();
        return result;
    }

    double wv = hierarchy->getNodeWeight(LEVEL, v) / hierarchy->getTotalContainedNodes(LEVEL, v);
    double wu = hierarchy->getNodeWeight(LEVEL, u) / hierarchy->getTotalContainedNodes(LEVEL, u);
    double edgeLength = idealEdgeLength(wv, wu);

    if (norm <= edgeLength) {
        result *= 0;
    } else {
        result *= options.embedderOptions.sigmoidScale / norm;
        result /= (double)hierarchy->getTotalContainedNodes(LEVEL, v);
    }

    if (!options.embedderOptions.neighborRepulsion) {
        result -= repulsionForce(v, LEVEL, u, LEVEL);
    }

    return result;
}

TmpCVec<1> LinearLayerEmbedder::repulsionForce(int v, int levelV, int u, int levelU) {
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

    double wv = hierarchy->getNodeWeight(levelV, v) / hierarchy->getTotalContainedNodes(levelV, v);
    double wu = hierarchy->getNodeWeight(levelU, u) / hierarchy->getTotalContainedNodes(levelU, u);
    double edgeLength = idealEdgeLength(wv, wu);

    if (norm > edgeLength) {
        result *= 0;
    } else {
        result *= options.embedderOptions.sigmoidScale / norm;
        result /= (double)hierarchy->getTotalContainedNodes(LEVEL, v);
    }
    return result;
}

double LinearLayerEmbedder::idealEdgeLength(double weightV, double weightU) {
    return options.embedderOptions.sigmoidLength * std::pow(weightU * weightV, 1.0 / options.dimension);
}
