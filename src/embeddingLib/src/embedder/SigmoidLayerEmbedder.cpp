#include "SigmoidLayerEmbedder.hpp"

TmpCVec<0> SigmoidLayerEmbedder::attractionForce(int v, int level, EdgeId target) {
    ASSERT(LEVEL == level);
    TmpVec<0> result(buffer, 0.0);

    NodeId u = graph.getEdgeTarget(target);
    result = hierarchy->getAveragePosition(level, u) - hierarchy->getAveragePosition(level, v);
    double norm = result.norm();

    if (norm <= 0) {
        LOG_WARNING("Random displacement attr V: (" << v << "," << level << ") U: (" << u << "," << level << ")");
        result.setToRandomUnitVector();
        return result;
    }

    double edgeWeight = hierarchy->getInverseEdgeWeightSum(level, target);
    int numEdges = hierarchy->getTotalContainedEdges(level, target);
    double averageEdgeWeight = edgeWeight / (double)numEdges;
    double sigmoidVal = sigmoid(options.embedderOptions.sigmoidScale * norm * averageEdgeWeight);

    double factor = 1.0;
    factor /= norm;
    factor /= (double)hierarchy->getTotalContainedNodes(LEVEL, v);
    factor *= (double)numEdges;
    factor *= sigmoidVal;
    factor *= averageEdgeWeight;

    result *= factor;
    return result;
}

TmpCVec<1> SigmoidLayerEmbedder::repulsionForce(int v, int levelV, int u, int levelU) {
    TmpVec<1> result(buffer, 0.0);

    if (v == u && levelV == levelU) return result;

    result = hierarchy->getAveragePosition(levelV, v) - hierarchy->getAveragePosition(levelU, u);
    double norm = result.norm();

    if (norm <= 0) {
        LOG_WARNING("Random displacement rep V: (" << v << "," << levelV << ") U: (" << u << "," << levelU << ")");
        result.setToRandomUnitVector();
        return result;
    }

    double weightV = hierarchy->getInverseScaledWeightSum(levelV, v);
    double weightU = hierarchy->getInverseScaledWeightSum(levelU, u);
    double totalWeight = weightU * weightV;

    int numEdges = hierarchy->getTotalContainedNodes(levelV, v) * hierarchy->getTotalContainedNodes(levelU, u);
    double averageEdgeWeight = totalWeight / (double)numEdges;
    double sigmoidVal = sigmoid(-options.embedderOptions.sigmoidScale * norm * averageEdgeWeight);

    double factor = 1.0;
    factor /= norm;
    factor /= (double)hierarchy->getTotalContainedNodes(levelV, v);
    factor *= (double)numEdges;
    factor *= sigmoidVal;
    factor *= averageEdgeWeight;

    result *= factor;
    return result;
}

double SigmoidLayerEmbedder::sigmoid(double x) {
    double result = 1.0 / (1 + std::exp(-x));
    ASSERT(result >= 0.0 && result <= 1.0);
    return result;
}

double SigmoidLayerEmbedder::invSigmoid(double x) {
    double result = 1.0 - sigmoid(x);
    ASSERT(result >= 0.0 && result <= 1.0);
    return result;
}