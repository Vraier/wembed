#include "WeightedSpringEmbedder.hpp"

void WeightedSpringEmbedder::initializeNewRun() {
    const int N = hierarchy->getLayerSize(LEVEL);
    currIteration = 0;
    weightSteps = 0;

    if (currentForce.size() == 0 && oldPositions.size() == 0) {
        currentForce.setSize(N);
        oldPositions.setSize(N);
    }

    // calculate lowestToCurrentMapping for attracting forces
    int lowestLevelSize = hierarchy->getLayerSize(0);
    lowestToCurrentMapping.resize(lowestLevelSize);

    for (NodeId v = 0; v < lowestLevelSize; v++) {
        int currLevel = 0;
        NodeId currParent = v;
        while (currLevel < LEVEL) {
            currParent = hierarchy->getParent(currLevel, currParent);
            currLevel++;
        }
        lowestToCurrentMapping[v] = currParent;
    }
}

void WeightedSpringEmbedder::calculateStep() {
    if (insignificantPosChange && !options.embedderOptions.staticWeights && (LEVEL == 0)) {
        insignificantPosChange = false;
        calculateWeightStep();
        weightSteps++;
    } else {
        calculateForceStep();
    }
    currIteration++;
}

void WeightedSpringEmbedder::calculateEmbedding() {
    LOG_INFO("Calculating layout...");
    initializeNewRun();
    while (!isFinished()) {
        calculateStep();
    }
    LOG_INFO("Finishes calculating layout");
}

bool WeightedSpringEmbedder::isFinished() {
    bool isFinished = (currIteration >= options.embedderOptions.maxIterations) || insignificantWeightChange ||
                      (LEVEL > 0 && insignificantPosChange) ||
                      (options.embedderOptions.staticWeights && insignificantPosChange);
    if (isFinished) {
        LOG_INFO("Finished in iteration " << currIteration << " with " << weightSteps << " weightSteps");
    }
    return isFinished;
}

void WeightedSpringEmbedder::calculateForceStep() {
    TmpVec<4> tmpVec(buffer, 0.0);
    const int N = hierarchy->getLayerSize(LEVEL);

    currentForce.setAll(0);
    oldPositions.setAll(0);

    // calculate new forces
    numRepForceCalculations = 0;
    calculateAllAttractingForces();
    calculateAllRepellingForces();
    double currCooling = std::pow(options.embedderOptions.coolingFactor, currIteration);
    for (NodeId v = 0; v < N; v++) {
        // cap the maximum replacement of the node
        currentForce[v].cWiseMax(-options.embedderOptions.maxDisplacement);
        currentForce[v].cWiseMin(options.embedderOptions.maxDisplacement);

        // apply cooling factor and speed
        currentForce[v] *= options.embedderOptions.speed * currCooling;
    }

    if (currIteration == 0) {
        DEBUG("Did " << (double)numRepForceCalculations / (double)N << " repulsion force calculations on average");
    }

    // apply cooling factor and movement based on force
    // also updates the position in the tree
    for (int v = 0; v < N; v++) {
        CVecRef vPos = hierarchy->getAveragePosition(LEVEL, v);
        oldPositions[v] = vPos;
        currentForce[v] += vPos;
        hierarchy->setPositionOfNode(LEVEL, v, currentForce[v]);
    }

    // calculate change in position
    double sumNormSquared = 0;
    double sumNormDiffSquared = 0;
    for (int v = 0; v < N; v++) {
        sumNormSquared += oldPositions[v].sqNorm();
        tmpVec = oldPositions[v] - hierarchy->getAveragePosition(LEVEL, v);
        sumNormDiffSquared += tmpVec.sqNorm();
    }
    if ((sumNormDiffSquared / sumNormSquared) < options.embedderOptions.relativePosMinChange) {
        insignificantPosChange = true;
    }
}

void WeightedSpringEmbedder::calculateWeightStep() {
    ASSERT(LEVEL == 0);

    const int N = hierarchy->getLayerSize(LEVEL);

    oldWeights.resize(N);
    newWeights.resize(N);
    double newWeightSum = 0;

    // calculate new weights
    for (NodeId a = 0; a < N; a++) {
        double wa = hierarchy->getNodeWeight(LEVEL, a);
        double idealWeight = getOptimalWeight(a);

        double idealThreshold = std::pow(idealWeight, 1.0 / options.dimension);
        double currThreshold = std::pow(wa, 1.0 / options.dimension);
        double newThreshold = currThreshold + std::pow(options.embedderOptions.weightCooling, weightSteps) *
                                                  options.embedderOptions.weightSpeed *
                                                  (idealThreshold - currThreshold);
        double newWeight = std::pow(
            newThreshold,
            options.dimension);  // NOTE(JP) i think this is worse; NOTE(JP) i think this is better for higher dimension
        // double newWeight = wa + std::pow(options.embedderOptions.weightCooling, weightSteps) *
        // options.embedderOptions.weightSpeed * (idealWeight - wa);

        oldWeights[a] = wa;
        newWeights[a] = newWeight;
        newWeightSum += newWeight;
    }

    // normalize weights and update them in graph
    // NOTE(JP) scaling the weights seem to perform better
    for (NodeId a = 0; a < N; a++) {
        newWeights[a] *= ((double)N / newWeightSum);
        hierarchy->setNodeWeight(LEVEL, a, newWeights[a]);
    }

    // calculate total change in weight
    double sumWeightDiff = 0;
    double sumNewWeights = 0;
    for (NodeId a = 0; a < N; a++) {
        sumWeightDiff += std::abs(newWeights[a] - oldWeights[a]);
        sumNewWeights += newWeights[a];
    }
    double relativeWeightChange = sumWeightDiff / sumNewWeights;

    if (relativeWeightChange < options.embedderOptions.relativeWeightMinChange) {
        LOG_DEBUG("Insignificant weight change of " << sumWeightDiff / sumNewWeights);
        insignificantWeightChange = true;
    }

    hierarchy->fixEdgeWeightsInLowestLayer();
}

void WeightedSpringEmbedder::calculateAllAttractingForces() {
    TmpVec<5> tempVec(buffer, 0.0);

    EmbeddedGraph& currGraph = hierarchy->graphs[LEVEL];
    const int N = currGraph.getNumVertices();

    for (NodeId a = 0; a < N; a++) {
        for (EdgeId e : currGraph.getEdges(a)) {
            NodeId b = currGraph.getEdgeTarget(e);
            tempVec = hierarchy->getAveragePosition(LEVEL, b) - hierarchy->getAveragePosition(LEVEL, a);
            double norm = tempVec.norm();

            if (norm > 0) {
                double edgeWeight = currGraph.getEdgeWeight(e);
                double desiredEdgeLength = 1.0 / edgeWeight;

                // we do not want the exact edge length
                if (options.embedderOptions.relaxedEdgeLength && (norm < desiredEdgeLength)) {
                    currentForce[a] += -1.0 * repulsionForce(a, LEVEL, b, LEVEL);
                }
                // we want the exact edge length
                else {
                    double factor = 1;
                    // NOTE(JP) this was an awkward and annoying bug <3 factor *= norm;
                    factor *= std::pow(norm, options.embedderOptions.forceExponent);
                    factor /= (double)hierarchy->getTotalContainedNodes(LEVEL, a);
                    factor *= edgeWeight / options.embedderOptions.cSpring;

                    tempVec *= 1.0 / norm;  // normalize the direction vector
                    tempVec *= factor;
                    currentForce[a] += tempVec;
                }
            } else {
                //  displace in random direction if positions are identical
                LOG_WARNING("Random displacement attr V: (" << a << "," << LEVEL << ") U: (" << b << "," << LEVEL
                                                            << ")");
                tempVec.setToRandomUnitVector();
                currentForce[a] += tempVec;
            }
        }
    }
}

void WeightedSpringEmbedder::calculateAllRepellingForces() {
    for (NodeId v = 0; v < hierarchy->getLayerSize(LEVEL); v++) {
        currentForce[v] += sumRepulsionForce(v);
    }
}

NodeId WeightedSpringEmbedder::lowestToCurrentLevel(NodeId v) {
    ASSERT(v < hierarchy->graphs[0].getNumVertices());
    return lowestToCurrentMapping[v];
}

TmpCVec<0> WeightedSpringEmbedder::sumRepulsionForce(int v) {
    TmpVec<0> sumRepForce(buffer, 0.0);

    // calculate repelling forces by moving up the partition tree, towards the root
    // take all the nodes that are children of the nodes on the path
    // only move up to maxApproxComparisons nodes up in the hierarchy
    int currParent = v;
    int currLevel = LEVEL;
    /*
    for (; currLevel < LEVEL + options.embedderOptions.maxApproxComparisons && currLevel < hierarchy->getNumLayers() -
    1; currLevel++) { int oldParent = currParent; currParent = hierarchy->getParent(currLevel, currParent); for (int
    child : hierarchy->getChildren(currLevel + 1, currParent)) { if (child == oldParent) continue;  // this node was
    already handled with greater precision sumRepForce += repulsionForce(v, LEVEL, child, currLevel);
        }
    }
    */

    int totalComparisons = 0;
    while (((totalComparisons + hierarchy->getLayerSize(currLevel)) > options.embedderOptions.maxApproxComparisons) &&
           currLevel < hierarchy->getNumLayers() - 1) {
        int oldParent = currParent;
        currParent = hierarchy->getParent(currLevel, currParent);
        for (int child : hierarchy->getChildren(currLevel + 1, currParent)) {
            if (child == oldParent) continue;  // this node was already handled with greater precision
            sumRepForce += repulsionForce(v, LEVEL, child, currLevel);
            totalComparisons++;
        }
        currLevel++;
    }

    // now handle all nodes on the current level (LEVEL+maxApproxComparisons)
    for (int coarseNode = 0; coarseNode < hierarchy->getLayerSize(currLevel); coarseNode++) {
        if (coarseNode == currParent) continue;  // this node was already handled with greater precision
        sumRepForce += repulsionForce(v, LEVEL, coarseNode, currLevel);
    }

    return sumRepForce;
}

TmpCVec<1> WeightedSpringEmbedder::repulsionForce(int v, int levelV, int u, int levelU) {
    TmpVec<1> result(buffer, 0.0);
    numRepForceCalculations++;

    if (v == u && levelV == levelU) return result;

    CVecRef posV = hierarchy->getAveragePosition(levelV, v);
    CVecRef posU = hierarchy->getAveragePosition(levelU, u);
    result = posV - posU;
    double norm = result.norm();

    // ensure positions are not identical
    if (norm > 0) {
        result *= (1.0 / norm);

        double weightV = hierarchy->getScaledWeightSum(levelV, v);
        double weightU = hierarchy->getScaledWeightSum(levelU, u);
        double factor = (weightV * weightU * options.embedderOptions.cSpring) /
                        (norm * hierarchy->getTotalContainedNodes(levelV, v));

        result *= factor;
        return result;
    } else {
        //  displace in random direction if positions are identical
        LOG_WARNING("Random displacement rep V: (" << v << "," << levelV << ") U: (" << u << "," << levelU << ")");
        result.setToRandomUnitVector();
        return result;
    }
}

double WeightedSpringEmbedder::idealEdgeLength(double wa, double wb) {
    return options.embedderOptions.cSpring * std::pow(wa * wb, 1.0 / (double)options.dimension);
    // return options.embedderOptions.cspring * std::pow(wa * wb, 1.0 / 2.0);
}

double WeightedSpringEmbedder::getOptimalWeight(NodeId a) {
    switch (options.embedderOptions.weightApproximation) {
        case 0:
            return getOptimalWeightUpperBound(a);
        case 1:
            return getOptimalWeightWeightedError(a);
        case 2:
            return getOptimalWeightExactError(a);
        case 3:
            return getOptimalWeightSampled(a);
        default:
            LOG_ERROR("Unknown weight approximation");
            return -1.0;
    }
}

double WeightedSpringEmbedder::getOptimalWeightUpperBound(NodeId a) {
    std::vector<std::pair<double, bool>> scaledWeights;
    addNeighborsToWeights(a, scaledWeights);

    // sort the vector by distance
    std::sort(scaledWeights.begin(), scaledWeights.end());

    double highestScaledDist = scaledWeights.back().first;
    return std::pow(highestScaledDist, options.dimension);
}

double WeightedSpringEmbedder::getOptimalWeightWeightedError(NodeId a) {
    std::vector<std::pair<double, bool>> scaledWeights;
    addAllNodesToWeights(a, scaledWeights);

    // sort the vector by distance
    std::sort(scaledWeights.begin(), scaledWeights.end());

    int bestIndex = getBestIndexRelative(a, scaledWeights);
    double bestThreshold = getBestThresholdForIndex(bestIndex, scaledWeights);
    return std::pow(bestThreshold, options.dimension);
}

double WeightedSpringEmbedder::getOptimalWeightExactError(NodeId a) {
    std::vector<std::pair<double, bool>> scaledWeights;
    addAllNodesToWeights(a, scaledWeights);

    // sort the vector by distance
    std::sort(scaledWeights.begin(), scaledWeights.end());

    int bestIndex = getBestIndexAbsolute(a, scaledWeights);
    double bestThreshold = getBestThresholdForIndex(bestIndex, scaledWeights);
    return std::pow(bestThreshold, options.dimension);
}

double WeightedSpringEmbedder::getOptimalWeightSampled(NodeId a) {
    TmpVec<6> tmpVec(buffer, 0.0);
    const EmbeddedGraph& currGraph = hierarchy->graphs[LEVEL];

    std::vector<std::pair<double, bool>> scaledWeights;
    addNeighborsToWeights(a, scaledWeights);

    // insert the sampled non neighbors
    for (std::pair<double, NodeId> p : hierarchy->getSampledNodeWeights(a)) {
        tmpVec = currGraph.getPosition(a) - currGraph.getPosition(p.second);
        double wb = p.first;
        double dist = tmpVec.norm();
        double scaledBDist = dist / std::pow(wb, 1.0 / (double)options.dimension);
        scaledWeights.push_back(std::make_pair(scaledBDist, false));
    }

    // sort the vector by distance
    std::sort(scaledWeights.begin(), scaledWeights.end());

    int bestIndex = getBestIndexRelative(a, scaledWeights);
    double bestThreshold = getBestThresholdForIndex(bestIndex, scaledWeights);
    return std::pow(bestThreshold, options.dimension);
}

void WeightedSpringEmbedder::addNeighborsToWeights(NodeId v, std::vector<std::pair<double, bool>>& weights) {
    TmpVec<7> tmpVec(buffer, 0.0);
    const EmbeddedGraph& currGraph = hierarchy->graphs[LEVEL];

    // insert the neighbors
    for (NodeId b : currGraph.getNeighbors(v)) {
        tmpVec = currGraph.getPosition(v) - currGraph.getPosition(b);
        double wb = currGraph.getNodeWeight(b);
        double dist = tmpVec.norm();
        ASSERT(dist > 0.0);
        double scaledBDist = dist / std::pow(wb, 1.0 / (double)options.dimension);
        weights.push_back(std::make_pair(scaledBDist, true));
    }
}

void WeightedSpringEmbedder::addAllNodesToWeights(NodeId v, std::vector<std::pair<double, bool>>& weights) {
    TmpVec<7> tmpVec(buffer, 0.0);
    const EmbeddedGraph& currGraph = hierarchy->graphs[LEVEL];
    const int N = currGraph.getNumVertices();
    weights.resize(N);

    // find the scaled distance to all other nodes
    for (NodeId b = 0; b < currGraph.getNumVertices(); b++) {
        tmpVec = currGraph.getPosition(v) - currGraph.getPosition(b);
        double wb = currGraph.getNodeWeight(b);
        double dist = tmpVec.norm();
        ASSERT(dist > 0.0);
        double scaledBDist = dist / std::pow(wb, 1.0 / (double)options.dimension);

        weights[b] = std::make_pair(scaledBDist, false);
    }

    // remember which nodes are neighbors
    for (NodeId b : currGraph.getNeighbors(v)) {
        weights[b].second = true;
    }
    weights[v].second = true;  // node is neighbor to itself
}

int WeightedSpringEmbedder::getBestIndexAbsolute(NodeId v, const std::vector<std::pair<double, bool>>& weights) const {
    // find the optimal index that minimizes the exact error
    int bestIndex = -1;
    const EmbeddedGraph& currGraph = hierarchy->graphs[LEVEL];
    double numWrongEdges = currGraph.getNumNeighbors(v);
    double numWrongNonEdges = weights.size() - numWrongEdges;
    double bestError = numWrongEdges + numWrongNonEdges;

    for (int i = 0; i < weights.size(); i++) {
        if (weights[i].second) {  // is a neighbor
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

    return bestIndex;
}

int WeightedSpringEmbedder::getBestIndexRelative(NodeId v, const std::vector<std::pair<double, bool>>& weights) const {
    // find the optimal index that minimizes the average error
    int bestIndex = -1;
    const EmbeddedGraph& currGraph = hierarchy->graphs[LEVEL];
    const double numEdges = currGraph.getNumNeighbors(v);
    const double numNonEdges = weights.size() - numEdges;
    double wrongEdges = 1;  // percent of how many edges are wrongly classified at the current index
    double wrongNonEdges = 0;
    double bestAverageError = wrongEdges + wrongNonEdges;

    for (int i = 0; i < weights.size(); i++) {
        if (weights[i].second) {  // is a neighbor
            wrongEdges -= 1.0 / numEdges;
        } else {  // is not a neighbor
            wrongNonEdges += 1.0 / numNonEdges;
        }
        double currAverageError = wrongEdges + wrongNonEdges;
        if (currAverageError < bestAverageError) {
            bestAverageError = currAverageError;
            bestIndex = i;
        }
    }

    return bestIndex;
}

double WeightedSpringEmbedder::getBestThresholdForIndex(int index,
                                                        const std::vector<std::pair<double, bool>>& weights) const {
    ASSERT(-1 <= index && index < (int)weights.size());
    double bestThreshold;
    if (index == -1) {
        // dont want to include any weight
        bestThreshold = weights[index + 1].first / 2.0;
    } else if (index == (int)weights.size() - 1) {
        // want to include all weights
        LOG_WARNING("Found best index to be last index. Dumping weights:");
        for (auto p : weights) {
            std::cout << "(" << p.first << ", " << p.second << ") ";
        }
        std::cout << "\n";
        bestThreshold = weights[index].first;
    } else {
        bestThreshold = (weights[index].first + weights[index + 1].first) / 2.0;
    }
    return bestThreshold;
}
