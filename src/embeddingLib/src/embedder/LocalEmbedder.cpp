#include "LocalEmbedder.hpp"

#include "SFMLDrawer.hpp"
#include "SVGDrawer.hpp"

void LocalEmbedder::calculateStep() {
    if (insignificantPosChange && !options.embedderOptions.staticWeights) {
        insignificantPosChange = false;
        calculateWeightStep();
    } else {
        calculateForceStep();
    }
    updateFurthestDistances();
    currIteration++;
}

TmpCVec<2> LocalEmbedder::repulsionForce(int v, int u) {
    TmpVec<2> result(buffer, 0.0);

    if (v == u)
        return result;

    CVecRef posV = graph.getPosition(v);
    CVecRef posU = graph.getPosition(u);
    result = posV - posU;
    double norm = result.norm();

    // ensure positions are not identical
    if (norm > 0) {
        result *= (1.0 / norm);  // normalize vector

        double currDist = norm / idealEdgeLength(graph.getNodeWeight(u), graph.getNodeWeight(v));
        double distDiff = currDist - maxNeighborDist[u];

        double factor = invSigmoid(options.embedderOptions.sigmoidScale * distDiff);

        result *= factor;
        return result;
    } else {
        //  displace in random direction if positions are identical
        LOG_WARNING( "Random displacement rep V: (" << v << ") U: (" << u << ")");
        result.setToRandomUnitVector();
        return result;
    }
}

TmpCVec<3> LocalEmbedder::attractionForce(int v, int u) {
    TmpVec<3> result(buffer, 0.0);
    if (v == u)
        return result;

    CVecRef posV = graph.getPosition(v);
    CVecRef posU = graph.getPosition(u);
    result = posU - posV;
    double norm = result.norm();

    if (norm > 0) {
        result *= (1.0 / norm);  // normalize vector

        double currDist = norm / idealEdgeLength(graph.getNodeWeight(u), graph.getNodeWeight(v));
        double distDiff = currDist - maxNeighborDist[u];

        double factor = sigmoid(options.embedderOptions.sigmoidScale * distDiff);

        result *= factor;

        result -= repulsionForce(v, u);  // cancel out repelling forces
        return result;

    } else {
        //  displace in random direction if positions are identical
        LOG_WARNING( "Random displacement attr V: (" << v << ") U: (" << u << ")");
        result.setToRandomUnitVector();
        return result;
    }
}

double LocalEmbedder::idealEdgeLength(double wa, double wb) {
    return options.embedderOptions.cSpring * std::pow(wa * wb, 1.0 / options.dimension);
}

double LocalEmbedder::sigmoid(double x) {
    return 1.0 / (1 + std::exp(-x));
}

double LocalEmbedder::invSigmoid(double x) {
    return sigmoid(-x);
}

void LocalEmbedder::updateFurthestDistances() {
    const int N = graph.getNumVertices();
    TmpVec<5> tmp(buffer, 0.0);

    for (NodeId v = 0; v < N; v++) {
        double maxDist = 0;

        for (NodeId u : graph.getNeighbors(v)) {
            CVecRef posV = graph.getPosition(v);
            CVecRef posU = graph.getPosition(u);
            tmp = posV - posU;

            double currDist = tmp.norm() / idealEdgeLength(graph.getNodeWeight(u), graph.getNodeWeight(v));
            maxDist = std::max(maxDist, currDist);
        }

        maxNeighborDist[v] = maxDist;
    }
}
