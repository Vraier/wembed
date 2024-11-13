#include "WeightedStressEmbedder.hpp"

#include <iostream>

void WeightedStressEmbedder::initializeNewRun() {
    const int N = hierarchy->getLayerSize(LEVEL);
    iterationCounter = 0;
    roundCounter = 0;

    if (newPositions.size() == 0 && oldPositions.size() == 0) {
        newPositions.setSize(N);
        oldPositions.setSize(N);
        rohs.resize(N);
    }

    // calculate lowestToCurrentMapping for attracting forces
    EmbeddedGraph& lowestGraph = hierarchy->graphs[0];
    lowestToCurrentMapping.resize(lowestGraph.getNumVertices());

    for (NodeId v = 0; v < lowestGraph.getNumVertices(); v++) {
        int currLevel = 0;
        NodeId currParent = v;
        while (currLevel < LEVEL) {
            currParent = hierarchy->getParent(currLevel, currParent);
            currLevel++;
        }
        lowestToCurrentMapping[v] = currParent;
    }
}

void WeightedStressEmbedder::calculateStep() {
    TmpVec<3> tmpVec(buffer, 0.0);
    const int N = hierarchy->getLayerSize(LEVEL);

    newPositions.setAll(0);
    oldPositions.setAll(0);
    std::fill(rohs.begin(), rohs.end(), 0);

    // calculate new forces
    numRepForceCalculations = 0;
    calculateAllAttractingForces();
    calculateAllRepellingForces();
    for (NodeId v = 0; v < N; v++) {
        newPositions[v] *= 1.0 / rohs[v];
    }

    if (roundCounter == 0 && iterationCounter == 0) {
        LOG_DEBUG("Did " << (double)numRepForceCalculations / (double)N << " repulsion force calculations on average");
    }

    // save old position and update position in the tree
    for (int v = 0; v < N; v++) {
        oldPositions[v] = hierarchy->getAveragePosition(LEVEL, v);  // Question(JP) is this dangerous?
        hierarchy->setPositionOfNode(LEVEL, v, newPositions[v]);
    }

    // calculate change in position
    double sumNormSquared = 0;
    double sumNormDiffSquared = 0;
    for (int v = 0; v < N; v++) {
        sumNormSquared += oldPositions[v].sqNorm();
        tmpVec = oldPositions[v] - newPositions[v];
        sumNormDiffSquared += tmpVec.sqNorm();
    }

    // LOG_DEBUG("Iteration= " << iterationCounter << " Round=" << roundCounter
    //                              << " Alpha=" << currentAlpha << " spooky=" << sumNormDiffSquared / sumNormSquared);

    // check if round is finished and new iteration should start
    roundCounter++;
    if ((roundCounter > options.embedderOptions.rounds) ||
        (sumNormDiffSquared / sumNormSquared) < options.embedderOptions.relativePosMinChange) {
        // start with the next round
        roundCounter = 0;
        iterationCounter++;
        currentAlpha =
            std::max(currentAlpha * options.embedderOptions.maxentAlphaCooling, options.embedderOptions.maxentMinAlpha);
    }
}

void WeightedStressEmbedder::calculateEmbedding() {
    LOG_INFO("Calculating layout...");
    initializeNewRun();
    while (!isFinished()) {
        calculateStep();
    }
    LOG_INFO("Finishes calculating layout");
}

bool WeightedStressEmbedder::isFinished() { return iterationCounter > options.embedderOptions.iterations; }

void WeightedStressEmbedder::calculateAllAttractingForces() {
    TmpVec<4> tempVec(buffer, 0.0);
    const EmbeddedGraph& lowestGraph = hierarchy->graphs[0];

    // iterate over all edges in the lowest layer
    for (NodeId a = 0; a < lowestGraph.getNumVertices(); a++) {
        for (EdgeId e : lowestGraph.getEdges(a)) {
            NodeId b = lowestGraph.getEdgeTarget(e);
            double wa = lowestGraph.getNodeWeight(a);
            double wb = lowestGraph.getNodeWeight(b);

            double clusterA = lowestToCurrentLevel(a);
            double clusterB = lowestToCurrentLevel(b);
            if (clusterA == clusterB) {
                // endpoints of edge are in same cluster
                // -> nothing to do
                continue;
            }

            CVecRef posA = hierarchy->getAveragePosition(LEVEL, clusterA);
            CVecRef posB = hierarchy->getAveragePosition(LEVEL, clusterB);
            tempVec = posA - posB;
            double norm = tempVec.norm();

            if (norm > 0) {
                // calculate weight factors and ideal edge length
                double duv = idealEdgeLenght(wa, wb);
                double wuv = 1.0 / (duv * duv);
                rohs[clusterA] += wuv;

                tempVec *= (duv / norm);
                tempVec += posB;
                tempVec *= wuv;
                newPositions[clusterA] += tempVec;
            } else {
                //  displace in random direction if positions are identical
                LOG_WARNING("Random displacement attr V: (" << clusterA << "," << LEVEL << ") U: (" << clusterB << ","
                                                            << LEVEL << ")");
                tempVec.setToRandomUnitVector();
                newPositions[clusterA] += tempVec;
            }

            // counteract the additional rep force calculations through the approximation
            tempVec = repulsionForce(clusterA, LEVEL, clusterB, LEVEL);
            tempVec *= 1.0 / hierarchy->getTotalContainedNodes(LEVEL, clusterB);
            newPositions[clusterA] -= tempVec;
        }
    }
}

void WeightedStressEmbedder::calculateAllRepellingForces() {
    for (NodeId v = 0; v < hierarchy->getLayerSize(LEVEL); v++) {
        newPositions[v] += sumRepulsionForce(v);
    }
}

NodeId WeightedStressEmbedder::lowestToCurrentLevel(NodeId v) {
    ASSERT(v < hierarchy->graphs[0].getNumVertices());
    return lowestToCurrentMapping[v];
}

TmpCVec<0> WeightedStressEmbedder::sumRepulsionForce(int v) {
    TmpVec<0> returnVal(buffer, 0.0);
    TmpVec<2> tmpVec(buffer, 0.0);

    // calculate repelling forces by moving up the partition tree, towards the root
    // take all the nodes that are children of the nodes on the path
    // only move up to maxApproxComparisons nodes up in the hierarchy
    int currParent = v;
    int currLevel = LEVEL;
    for (;
         currLevel < LEVEL + options.embedderOptions.maxApproxComparisons && currLevel < hierarchy->getNumLayers() - 1;
         currLevel++) {
        int oldParent = currParent;
        currParent = hierarchy->getParent(currLevel, currParent);
        for (int child : hierarchy->getChildren(currLevel + 1, currParent)) {
            if (child == oldParent) continue;  // this node was already handled with greater precision
            returnVal += repulsionForce(v, LEVEL, child, currLevel);
            numRepForceCalculations++;
        }
    }

    // now handle all nodes on the current level (LEVEL+maxApproxComparisons)
    for (int coarseNode = 0; coarseNode < hierarchy->getLayerSize(currLevel); coarseNode++) {
        if (coarseNode == currParent) continue;  // this node was already handle with greater precision
        returnVal += repulsionForce(v, LEVEL, coarseNode, currLevel);
    }
    return returnVal;
}

TmpCVec<1> WeightedStressEmbedder::repulsionForce(int v, int levelV, int u, int levelU) {
    TmpVec<1> returnVal(buffer, 0.0);

    if (v == u && levelV == levelU) return returnVal;

    CVecRef posV = hierarchy->getAveragePosition(levelV, v);
    CVecRef posU = hierarchy->getAveragePosition(levelU, u);
    returnVal = posV - posU;
    double sqNorm = returnVal.sqNorm();

    // ensure position is not identical
    if (sqNorm > 0) {
        returnVal *= (1.0 / sqNorm);
        // scale repulsion force by the size of the children
        returnVal *= hierarchy->getTotalContainedNodes(levelU, u);
        returnVal *= currentAlpha;
        return returnVal;
    } else {
        // displace in random direction if positions are identical
        LOG_WARNING("Random displacement rep V: (" << v << "," << levelV << ") U: (" << u << "," << levelU << ")");
        returnVal.setToRandomUnitVector();
        returnVal *= currentAlpha;
        return returnVal;
    }
}

double WeightedStressEmbedder::idealEdgeLenght(double wa, double wb) {
    return options.embedderOptions.cSpring * std::pow(wa * wb, 1.0 / (double)options.dimension);
}
