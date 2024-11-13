#include "AbstractLayerEmbedder.hpp"

#include <queue>

void AbstractSimpleEmbedder::initializeNewRun() {
    const int N = hierarchy->getLayerSize(LEVEL);
    numIterations = 0;
    numForceSteps = 0;
    numWeightSteps = 0;

    if (currentForce.size() == 0 && oldPositions.size() == 0) {
        currentForce.setSize(N);
        oldPositions.setSize(N);
    }
}

void AbstractSimpleEmbedder::calculateEmbedding() {
    LOG_INFO("Calculating layout...");

    initializeNewRun();  // TODO: make this consistent with AbstractSimpleEmbedder
    while (!isFinished()) {
        calculateStep();
    }
    LOG_INFO("Finished in iteration " << numIterations << " with " << numForceSteps << " forceSteps and "
                                      << numWeightSteps << " weightSteps");
}

void AbstractSimpleEmbedder::calculateStep() {
    if (insignificantPosChange && !options.embedderOptions.staticWeights && (LEVEL == 0)) {
        insignificantPosChange = false;
        calculateWeightStep();
    } else {
        calculateForceStep();
    }

    numIterations++;
}

bool AbstractSimpleEmbedder::isFinished() {
    bool isFinished = (numIterations >= options.embedderOptions.maxIterations) || insignificantWeightChange ||
                      (LEVEL > 0 && insignificantPosChange) ||
                      (options.embedderOptions.staticWeights && insignificantPosChange);
    return isFinished;
}

void AbstractSimpleEmbedder::calculateForceStep() {
    numForceSteps++;
    TmpVec<FORCE_STEP_BUFFER> tmpVec(buffer, 0.0);
    const int N = hierarchy->getLayerSize(LEVEL);
    ASSERT(N == graph.getNumVertices());

    currentForce.setAll(0);
    oldPositions.setAll(0);

    // calculate new forces. this fills currentForce
    numRepForceCalculations = 0;
    calculateAllAttractingForces();
    calculateAllRepellingForces();

    // apply cooling and speed
    double currCooling = std::pow(options.embedderOptions.coolingFactor, numIterations);
    for (NodeId v = 0; v < N; v++) {
        // cap the maximum replacement of the node
        currentForce[v].cWiseMax(-options.embedderOptions.maxDisplacement);
        currentForce[v].cWiseMin(options.embedderOptions.maxDisplacement);

        // apply cooling factor and speed
        currentForce[v] *= options.embedderOptions.speed * currCooling;
    }

    if (numIterations == 0) {
        LOG_DEBUG("Did " << (double)numRepForceCalculations / (double)N << " repulsion force calculations on average");
    }

    // apply movement based on force
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

void AbstractSimpleEmbedder::calculateWeightStep() {
    ASSERT(LEVEL == 0);
    numWeightSteps++;

    // TODO
    return;
}

void AbstractSimpleEmbedder::calculateAllAttractingForces() {
    const int N = graph.getNumVertices();

    for (NodeId a = 0; a < N; a++) {
        for (EdgeId e : graph.getEdges(a)) {
            currentForce[a] += attractionForce(a, LEVEL, e);
        }
    }
}

void AbstractSimpleEmbedder::calculateAllRepellingForces() {
    if (options.embedderOptions.approxSelectionType == 0) {
        calculateAllRepellingForcesCoarse();
    } else {
        calculateAllRepellingForcesPriority();
    }
}

void AbstractSimpleEmbedder::calculateAllRepellingForcesCoarse() {
    TmpVec<ALL_REP_BUFFER> sumRepForce(buffer, 0.0);

    // calculate repelling forces by moving up the partition tree, towards the root
    // take all the nodes that are children of the nodes on the path
    for (NodeId v = 0; v < hierarchy->getLayerSize(LEVEL); v++) {
        int currParent = v;
        int oldParent = -1;
        int currLevel = LEVEL;
        sumRepForce.setAll(0);  // bugggg was here

        // try to compare only up to maxApproxComparisons nodes in the hierarchy
        int totalComparisons = 0;
        while ((totalComparisons + hierarchy->getLayerSize(currLevel) > options.embedderOptions.maxApproxComparisons) &&
               currLevel < hierarchy->getNumLayers() - 1) {
            oldParent = currParent;
            currParent = hierarchy->getParent(currLevel, currParent);
            for (int child : hierarchy->getChildren(currLevel + 1, currParent)) {
                if (child == oldParent) continue;  // this node was already handled with greater precision
                sumRepForce += repulsionForce(v, LEVEL, child, currLevel);
                totalComparisons++;
                numRepForceCalculations++;
            }
            currLevel++;
        }

        // now handle all nodes on the current level (LEVEL+maxApproxComparisons)
        for (int coarseNode = 0; coarseNode < hierarchy->getLayerSize(currLevel); coarseNode++) {
            if (coarseNode == currParent) continue;  // this node was already handled with greater precision
            sumRepForce += repulsionForce(v, LEVEL, coarseNode, currLevel);
            numRepForceCalculations++;
        }

        currentForce[v] += sumRepForce;
    }
}

void AbstractSimpleEmbedder::calculateAllRepellingForcesPriority() {
    TmpVec<ALL_REP_BUFFER> sumRepForce(buffer, 0.0);
    TmpVec<REP_AUX_BUFFER> tmpVec(buffer, 0.0);

    // min priority queue that maps distance to level and clusterID
    using pq_element = std::pair<double, std::pair<int, int>>;
    std::priority_queue<pq_element, std::vector<pq_element>, std::greater<pq_element>> pq;
    std::vector<pq_element> lowestLevel;  // stores all nodes in lowest level (because they can't be expanded)

    for (NodeId v = 0; v < hierarchy->getLayerSize(LEVEL); v++) {
        sumRepForce.setAll(0);
        pq = std::priority_queue<pq_element, std::vector<pq_element>,
                                 std::greater<pq_element>>();  // clear priority queue
        lowestLevel.clear();
        CVecRef vPos = hierarchy->getAveragePosition(LEVEL, v);

        int currLevel = LEVEL;
        int currNode = v;
        int currParent = hierarchy->getParent(currLevel, currNode);

        // insert all nodes of path from v to root into pq
        while (currLevel < hierarchy->getNumLayers() - 1) {
            for (int child : hierarchy->getChildren(currLevel + 1, currParent)) {
                if (child == currNode) continue;  // this node was already handled with greater precision

                tmpVec = vPos - hierarchy->getAveragePosition(currLevel, child);
                double dist = tmpVec.norm();

                if (currLevel == LEVEL) {
                    lowestLevel.push_back({dist, {currLevel, child}});
                } else {
                    pq.push({dist, {currLevel, child}});
                }
            }
            currNode = currParent;
            currLevel++;
            currParent = hierarchy->getParent(currLevel, currParent);
        }

        // expand the pq as long as we haven't found enough nodes for repelling forces
        while (pq.size() + lowestLevel.size() < options.embedderOptions.maxApproxComparisons && !pq.empty()) {
            auto [dist, levelAndChild] = pq.top();
            pq.pop();
            auto [currLevel, child] = levelAndChild;
            int grandChildLevel = currLevel - 1;

            // insert all children of the child into the pq
            for (int grandChild : hierarchy->getChildren(currLevel, child)) {
                tmpVec = vPos - hierarchy->getAveragePosition(grandChildLevel, grandChild);
                double dist = tmpVec.norm();

                if (grandChildLevel == LEVEL) {
                    lowestLevel.push_back({dist, {grandChildLevel, grandChild}});
                } else {
                    pq.push({dist, {grandChildLevel, grandChild}});
                }
            }
        }

        // sum up the repulsion forces from the nodes in the pq and lowest level
        while (!pq.empty()) {
            auto [dist, levelAndChild] = pq.top();
            pq.pop();
            auto [currLevel, child] = levelAndChild;
            sumRepForce += repulsionForce(v, LEVEL, child, currLevel);
            numRepForceCalculations++;
        }
        for (auto [dist, levelAndChild] : lowestLevel) {
            auto [currLevel, child] = levelAndChild;
            sumRepForce += repulsionForce(v, LEVEL, child, currLevel);
            numRepForceCalculations++;
        }

        currentForce[v] += sumRepForce;
    }
}
