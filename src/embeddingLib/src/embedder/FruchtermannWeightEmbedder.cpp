#include "FruchtermannWeightEmbedder.hpp"

TmpCVec<2> FruchtermannWeightEmbedder::repulsionForce(int v, int u) {
    TmpVec<REP_BUFFER> result(buffer, 0.0);

    if (v == u)
        return result;

    CVecRef posV = graph.getPosition(v);
    CVecRef posU = graph.getPosition(u);
    result = posV - posU;
    double norm = result.norm();

    // ensure positions are not identical
    if (norm > 0) {
        result *= (1.0 / norm);  // normalize vector

        double desiredEdgeLength = idealEdgeLength(graph.getNodeWeight(u), graph.getNodeWeight(v));
        double factor = std::pow(desiredEdgeLength, options.embedderOptions.forceExponent) / norm;
        result *= factor;
        return result;
    } else {
        //  displace in random direction if positions are identical
        LOG_WARNING( "Random displacement rep V: (" << v << ") U: (" << u << ")");
        result.setToRandomUnitVector();
        return result;
    }
}

TmpCVec<3> FruchtermannWeightEmbedder::attractionForce(int v, int u) {
    TmpVec<ATTR_BUFFER> result(buffer, 0.0);
    if (v == u)
        return result;

    CVecRef posV = graph.getPosition(v);
    CVecRef posU = graph.getPosition(u);
    result = posU - posV;
    double norm = result.norm();

    if (norm > 0) {
        result *= (1.0 / norm);  // normalize vector

        double desiredEdgeLength = idealEdgeLength(graph.getNodeWeight(u), graph.getNodeWeight(v));

        // we do not want the exact edge length
        if (options.embedderOptions.relaxedEdgeLength && (norm < desiredEdgeLength)) {
            result = -1.0 * repulsionForce(v, u);
            return result;
        }
        // we want the exact edge length
        else {
            double factor = std::pow(norm, options.embedderOptions.forceExponent) / desiredEdgeLength;
            result *= factor;
            return result;
        }
    } else {
        //  displace in random direction if positions are identical
        LOG_WARNING( "Random displacement attr V: (" << v << ") U: (" << u << ")");
        result.setToRandomUnitVector();
        return result;
    }
}

double FruchtermannWeightEmbedder::idealEdgeLength(double wa, double wb) {
    return options.embedderOptions.cSpring * std::pow(wa * wb, 1.0 / options.dimension);
}

void FruchtermannWeightEmbedder::calculateWeightStep() {
    const int N = graph.getNumVertices();

    double newWeightSum = 0;

    // calculate new weights
    for (NodeId a = 0; a < N; a++) {
        double wa = graph.getNodeWeight(a);
        double idealWeight = getOptimalWeight(a);
        double newWeight = wa + options.embedderOptions.weightSpeed * (idealWeight - wa);

        oldWeights[a] = wa;
        newWeights[a] = newWeight;
        newWeightSum += newWeight;
    }

    // normalize weights and update them in graph
    for (NodeId a = 0; a < N; a++) {
        newWeights[a] *= ((double)N / newWeightSum);
        // NOTE(JP) scaling the weights seem to perform better
        graph.setNodeWeight(a, newWeights[a]);
    }

    double sumWeightDiff = 0;
    double sumNewWeights = 0;
    for (NodeId a = 0; a < N; a++) {
        sumWeightDiff += std::abs(newWeights[a] - oldWeights[a]);
        sumNewWeights += newWeights[a];
    }
    double relativeWeightChange = sumWeightDiff / sumNewWeights;

    if (relativeWeightChange < options.embedderOptions.relativeWeightMinChange) {
        // LOG_DEBUG( "Insignificant weight change of " << sumWeightDiff / sumNewWeights);
        insignificantWeightChange = true;
    }
}

double FruchtermannWeightEmbedder::getOptimalWeight(NodeId a) {
    switch (options.embedderOptions.weightApproximation) {
        case 0:
            return getOptimalWeightUpperBound(a);
        case 1:
            return getOptimalWeightWeightedError(a);
        case 2:
            return getOptimalWeightExactError(a);
        default:
            LOG_ERROR( "Unknown weight approximation");
            return -1.0;
    }
}

double FruchtermannWeightEmbedder::getOptimalWeightUpperBound(NodeId a) {
    TmpVec<WEIGHT_BUFFER> tmpVec(buffer, 0.0);

    double highestScaledDist = -1;

    for (NodeId b : graph.getNeighbors(a)) {
        tmpVec = graph.getPosition(a) - graph.getPosition(b);
        double wb = graph.getNodeWeight(b);
        double dist = tmpVec.norm();
        double scaledBDist = dist / std::pow(wb, 1.0 / (double)options.dimension);

        highestScaledDist = std::max(highestScaledDist, scaledBDist);
    }

    return std::pow(highestScaledDist, options.dimension);
}

double FruchtermannWeightEmbedder::getOptimalWeightWeightedError(NodeId a) {
    TmpVec<WEIGHT_BUFFER> tmpVec(buffer, 0.0);
    const int N = graph.getNumVertices();

    std::vector<std::pair<double, bool>> scaledWeights(N);

    // find the scaled distance to all other nodes
    for (NodeId b = 0; b < N; b++) {
        tmpVec = graph.getPosition(a) - graph.getPosition(b);
        double wb = graph.getNodeWeight(b);
        double dist = tmpVec.norm();
        double scaledBDist = dist / std::pow(wb, 1.0 / (double)options.dimension);

        scaledWeights[b] = std::make_pair(scaledBDist, false);
    }
    // remember which nodes are neighbors
    for (NodeId b : graph.getNeighbors(a)) {
        scaledWeights[b].second = true;
    }
    scaledWeights[a].second = true;  // node is neighbor to itself

    // sort the vector by distance
    std::sort(scaledWeights.begin(), scaledWeights.end());

    // find the optimal index that minimizes the average error
    const double numEdges = graph.getNumNeighbors(a) + 1;
    const double numNonEdges = N - numEdges;
    double wrongEdges = 1;  // percent of how many edges are wrongly classified at the current index
    double wrongNonEdges = 0;
    double bestAverageError = (wrongEdges + wrongNonEdges) / 2.0;
    int bestIndex = -1;
    for (int i = 0; i < N; i++) {
        if (scaledWeights[i].second) {  // is a neighbor
            wrongEdges -= 1.0 / numEdges;
        } else {  // is not a neighbor
            wrongNonEdges += 1.0 / numNonEdges;
        }
        double currAverageError = (wrongEdges + wrongNonEdges) / 2.0;

        if (currAverageError < bestAverageError) {
            bestAverageError = currAverageError;
            bestIndex = i;
        }
    }

    ASSERT(bestIndex > 0 && bestIndex < N);  // TODO(JP) maybe do something sensible in this edge case
    double bestThreshold = (scaledWeights[bestIndex].first + scaledWeights[bestIndex + 1].first) / 2.0;

    return std::pow(bestThreshold, options.dimension);
}

double FruchtermannWeightEmbedder::getOptimalWeightExactError(NodeId a) {
    TmpVec<WEIGHT_BUFFER> tmpVec(buffer, 0.0);
    const int N = graph.getNumVertices();

    std::vector<std::pair<double, bool>> scaledWeights(N);

    // find the scaled distance to all other nodes
    for (NodeId b = 0; b < N; b++) {
        tmpVec = graph.getPosition(a) - graph.getPosition(b);
        double wb = graph.getNodeWeight(b);
        double dist = tmpVec.norm();
        double scaledBDist = dist / std::pow(wb, 1.0 / (double)options.dimension);

        scaledWeights[b] = std::make_pair(scaledBDist, false);
    }
    // remember which nodes are neighbors
    for (NodeId b : graph.getNeighbors(a)) {
        scaledWeights[b].second = true;
    }
    scaledWeights[a].second = true;  // node is neighbor to itself

    // sort the vector by distance
    std::sort(scaledWeights.begin(), scaledWeights.end());

    // find the optimal index that minimizes the average error
    double numWrongEdges = graph.getNumNeighbors(a) + 1;
    double numWrongNonEdges = N - numWrongEdges;
    double bestError = numWrongEdges + numWrongNonEdges;
    int bestIndex = -1;
    for (int i = 0; i < N; i++) {
        if (scaledWeights[i].second) {  // is a neighbor
            numWrongEdges -= 1.0;
        } else {  // is not a neighbor
            numWrongNonEdges += 1.0;
        }
        double currError = numWrongEdges + numWrongNonEdges;

        if (currError < bestError) {
            bestError = currError;
            bestIndex = i;
        }
    }

    ASSERT(bestIndex > 0 && bestIndex < N);  // TODO(JP) maybe do something sensible in this edge case
    double bestThreshold = (scaledWeights[bestIndex].first + scaledWeights[bestIndex + 1].first) / 2.0;

    return std::pow(bestThreshold, options.dimension);
}
