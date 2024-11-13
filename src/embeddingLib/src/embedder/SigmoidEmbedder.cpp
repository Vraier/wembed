#include "SigmoidEmbedder.hpp"

#include "SFMLDrawer.hpp"
#include "SVGDrawer.hpp"

TmpCVec<2> SigmoidEmbedder::repulsionForce(int v, int u) {
    TmpVec<2> result(buffer, 0.0);

    if (v == u) return result;

    CVecRef posV = graph.getPosition(v);
    CVecRef posU = graph.getPosition(u);
    double wU = graph.getNodeWeight(u);
    double wV = graph.getNodeWeight(v);

    result = posV - posU;
    double norm = result.norm();

    if (norm <= 0) {
        //  displace in random direction if positions are identical
        LOG_WARNING( "Random displacement rep V: (" << v << ") U: (" << u << ")");
        result.setToRandomUnitVector();
        return result;
    }

    double sigmoidVal = norm / idealEdgeLength(wU, wV);
    double sigm = sigmoid(-sigmoidVal * options.embedderOptions.sigmoidScale);  // negative sign is important here!

    double factor = sigm / (wU * wV * norm);

    result *= factor;
    return result;
}

TmpCVec<3> SigmoidEmbedder::attractionForce(int v, int u) {
    TmpVec<3> result(buffer, 0.0);
    if (v == u) return result;

    CVecRef posV = graph.getPosition(v);
    CVecRef posU = graph.getPosition(u);
    double wU = graph.getNodeWeight(u);
    double wV = graph.getNodeWeight(v);

    result = posU - posV;
    double norm = result.norm();

    if (norm <= 0) {
        //  displace in random direction if positions are identical
        LOG_WARNING( "Random displacement attr V: (" << v << ") U: (" << u << ")");
        result.setToRandomUnitVector();
        return result;
    }

    double sigmoidVal = norm / idealEdgeLength(wU, wV);
    double sigm = sigmoid(sigmoidVal * options.embedderOptions.sigmoidScale);  // no negative sign here!

    double factor = sigm / (wU * wV * norm);

    result *= factor;
    return result;
}

double SigmoidEmbedder::weightRepulsionForce(NodeId v, NodeId u) {
    TmpVec<3> tmpVec(buffer, 0.0);
    CVecRef posV = graph.getPosition(v);
    CVecRef posU = graph.getPosition(u);
    double wU = graph.getNodeWeight(u);
    double wV = graph.getNodeWeight(v);

    tmpVec = posU - posV;
    double norm = tmpVec.norm();
    double sigmoidVal = norm / idealEdgeLength(wU, wV);

    double weightForceScale = norm / ((double)options.dimension * std::pow(wV, 1.0 / options.dimension) *
                                      std::pow(wU, 1.0 + 1.0 / options.dimension));

    return sigmoid(-options.embedderOptions.sigmoidScale * sigmoidVal) * weightForceScale;
}

double SigmoidEmbedder::weightAttractionForce(NodeId v, NodeId u) {
    TmpVec<3> tmpVec(buffer, 0.0);
    CVecRef posV = graph.getPosition(v);
    CVecRef posU = graph.getPosition(u);
    double wU = graph.getNodeWeight(u);
    double wV = graph.getNodeWeight(v);

    tmpVec = posU - posV;
    double norm = tmpVec.norm();
    double sigmoidVal = norm / idealEdgeLength(wU, wV);

    double weightForceScale = norm / ((double)options.dimension * std::pow(wV, 1.0 / options.dimension) *
                                      std::pow(wU, 1.0 + 1.0 / options.dimension));
    
    return sigmoid(options.embedderOptions.sigmoidScale * sigmoidVal) * weightForceScale - weightRepulsionForce(v, u);
}

double SigmoidEmbedder::idealEdgeLength(double wa, double wb) {
    return options.embedderOptions.cSpring * std::pow(wa * wb, 1.0 / options.dimension);
}

double SigmoidEmbedder::sigmoid(double x) { return 1.0 / (1 + std::exp(-x)); }