#include "SimpleEmbedders.hpp"

#include <fstream>

TmpCVec<2> SimpleF2VEmbedder::repulsionForce(int v, int u) {
    TmpVec<2> result(buffer, 0.0);

    if (v == u) return result;

    CVecRef posV = graph.getPosition(v);
    CVecRef posU = graph.getPosition(u);

    double dotP = dotProduct(posV, posU);

    result = posU;
    result *= -sigmoid(dotP);
    return result;
}

TmpCVec<3> SimpleF2VEmbedder::attractionForce(int v, int u) {
    TmpVec<3> result(buffer, 0.0);
    if (v == u) return result;

    CVecRef posV = graph.getPosition(v);
    CVecRef posU = graph.getPosition(u);

    double dotP = dotProduct(posV, posU);
    result = posU;
    result *= sigmoid(-dotP);
    return result;
}

void SimpleF2VEmbedder::initializeCoordinates() {
    const int N = graph.getNumVertices();
    for (int i = 0; i < N; i++) {
        // use random initial coords in unit cube
        for (int j = 0; j < options.embeddingDimension; j++) {
            graph.coordinates[i][j] = Rand::randomDouble(0, 1);
        }
    }
}

double SimpleF2VEmbedder::sigmoid(double x) { return 1.0 / (1 + std::exp(-x)); }

double SimpleF2VEmbedder::dotProduct(CVecRef a, CVecRef b) {
    ASSERT(a.dimension() == b.dimension(), "Dimensions of vectors do not match!");
    double result = 0.0;
    for (int i = 0; i < a.dimension(); i++) {
        result += a[i] * b[i];
    }
    return result;
}

TmpCVec<2> SimpleF2VNormedEmbedder::repulsionForce(int v, int u) {
    TmpVec<2> result(buffer, 0.0);

    if (v == u) return result;

    CVecRef posV = graph.getPosition(v);
    CVecRef posU = graph.getPosition(u);

    double dotP = dotProduct(posV, posU);

    result = posU;
    result *= -sigmoid(dotP);
    return result;
}

TmpCVec<3> SimpleF2VNormedEmbedder::attractionForce(int v, int u) {
    TmpVec<3> result(buffer, 0.0);
    if (v == u) return result;

    CVecRef posV = graph.getPosition(v);
    CVecRef posU = graph.getPosition(u);

    double dotP = dotProduct(posV, posU);
    result = posU;
    result *= sigmoid(-dotP);
    return result;
}

void SimpleF2VNormedEmbedder::calculateForceStep() {
    numForceSteps++;
    TmpVec<FORCE_STEP_BUFFER> tmpVec(buffer, 0.0);
    const int N = graph.getNumVertices();

    currentForce.setAll(0);
    oldPositions.setAll(0);

    // calculate new forces
    calculateAllAttractingForces();
    calculateAllRepellingForces();
    double currCooling = std::pow(options.coolingFactor, currIteration);
    for (int v = 0; v < N; v++) {
        currentForce[v] *= options.speed * currCooling;
    }

    // apply movement based on force
    for (int v = 0; v < N; v++) {
        oldPositions[v] = graph.coordinates[v];
        graph.coordinates[v] += currentForce[v];
    }

    // normalize the positions
    for (int v = 0; v < N; v++) {
        if (graph.coordinates[v].norm() <= 0) {
            graph.coordinates[v].setToRandomUnitVector();
        } else {
            graph.coordinates[v] /= graph.coordinates[v].norm();
        }
    }

    // calculate change in position
    double sumNormSquared = 0;
    double sumNormDiffSquared = 0;
    for (int v = 0; v < N; v++) {
        sumNormSquared += oldPositions[v].sqNorm();
        tmpVec = oldPositions[v] - graph.coordinates[v];
        sumNormDiffSquared += tmpVec.sqNorm();
    }
    if ((sumNormDiffSquared / sumNormSquared) < options.relativePosMinChange) {
        insignificantPosChange = true;
    }
}

void SimpleF2VNormedEmbedder::initializeCoordinates() {
    const int N = graph.getNumVertices();
    for (int i = 0; i < N; i++) {
        // use random initial coords in unit cube
        for (int j = 0; j < options.embeddingDimension; j++) {
            graph.coordinates[i][j] = Rand::randomDouble(0, 1);
        }
    }
}

double SimpleF2VNormedEmbedder::sigmoid(double x) { return 1.0 / (1 + std::exp(-x)); }

double SimpleF2VNormedEmbedder::dotProduct(CVecRef a, CVecRef b) {
    ASSERT(a.dimension() == b.dimension(), "Dimensions of vectors do not match!");
    double result = 0.0;
    for (int i = 0; i < a.dimension(); i++) {
        result += a[i] * b[i];
    }
    return result;
}

TmpCVec<2> SimpleFruchtermannEmbedder::repulsionForce(int v, int u) {
    TmpVec<2> result(buffer, 0.0);

    if (v == u) return result;

    CVecRef posV = graph.getPosition(v);
    CVecRef posU = graph.getPosition(u);

    result = posV - posU;
    double norm = result.norm();

    if (norm <= 0) {
        LOG_WARNING("Random displacement rep V: (" << v << ") U: (" << u << ")");
        result.setToRandomUnitVector();
        return result;
    }

    result /= norm * norm;
    return result;
}

TmpCVec<3> SimpleFruchtermannEmbedder::attractionForce(int v, int u) {
    TmpVec<3> result(buffer, 0.0);
    if (v == u) return result;

    CVecRef posV = graph.getPosition(v);
    CVecRef posU = graph.getPosition(u);

    result = posU - posV;
    double norm = result.norm();

    if (norm <= 0) {
        LOG_WARNING("Random displacement attr V: (" << v << ") U: (" << u << ")");
        result.setToRandomUnitVector();
        return result;
    }

    result *= norm;
    return result;
}

TmpCVec<2> SimpleMaxentEmbedder::repulsionForce(int v, int u) {
    TmpVec<2> result(buffer, 0.0);

    if (v == u) return result;

    CVecRef posV = graph.getPosition(v);
    CVecRef posU = graph.getPosition(u);

    result = posV - posU;
    double norm = result.norm();

    if (norm <= 0) {
        LOG_WARNING("Random displacement rep V: (" << v << ") U: (" << u << ")");
        result.setToRandomUnitVector();
        return result;
    }

    result /= norm * norm;
    return result;
}

TmpCVec<3> SimpleMaxentEmbedder::attractionForce(int v, int u) {
    TmpVec<3> result(buffer, 0.0);
    if (v == u) return result;

    CVecRef posV = graph.getPosition(v);
    CVecRef posU = graph.getPosition(u);

    result = posU - posV;
    double norm = result.norm();

    if (norm <= 0) {
        LOG_WARNING("Random displacement rep V: (" << v << ") U: (" << u << ")");
        result.setToRandomUnitVector();
        return result;
    }

    result *= 2.0 * (norm - 1.0);
    return result;
}

TmpCVec<2> SimpleSigmoidEuclideanEmbedder::repulsionForce(int v, int u) {
    TmpVec<2> result(buffer, 0.0);

    if (v == u) return result;

    CVecRef posV = graph.getPosition(v);
    CVecRef posU = graph.getPosition(u);

    result = posV - posU;
    double norm = result.norm();

    if (norm <= 0) {
        //  displace in random direction if positions are identical
        LOG_WARNING("Random displacement rep V: (" << v << ") U: (" << u << ")");
        result.setToRandomUnitVector();
        return result;
    }

    result /= norm;
    result *= options.sigmoidScale;
    result *= sigmoid(options.sigmoidScale * (options.sigmoidLength - norm));
    return result;
}

TmpCVec<3> SimpleSigmoidEuclideanEmbedder::attractionForce(int v, int u) {
    TmpVec<3> result(buffer, 0.0);
    if (v == u) return result;

    CVecRef posV = graph.getPosition(v);
    CVecRef posU = graph.getPosition(u);

    result = posU - posV;
    double norm = result.norm();

    if (norm <= 0) {
        //  displace in random direction if positions are identical
        LOG_WARNING("Random displacement rep V: (" << v << ") U: (" << u << ")");
        result.setToRandomUnitVector();
        return result;
    }

    result /= norm;
    result *= options.sigmoidScale;
    result *= sigmoid(options.sigmoidScale * (norm - options.sigmoidLength));
    return result;
}

double SimpleSigmoidEuclideanEmbedder::sigmoid(double x) { return 1.0 / (1 + std::exp(-x)); }

TmpCVec<2> SimpleSigmoidNoDimEmbedder::repulsionForce(int v, int u) {
    TmpVec<2> result(buffer, 0.0);

    if (v == u) return result;

    CVecRef posV = graph.getPosition(v);
    CVecRef posU = graph.getPosition(u);

    result = posV - posU;
    double norm = result.norm();

    if (norm <= 0) {
        //  displace in random direction if positions are identical
        LOG_WARNING("Random displacement rep V: (" << v << ") U: (" << u << ")");
        result.setToRandomUnitVector();
        return result;
    }

    double weightFactor = idealEdgeLength(graph.getNodeWeight(v), graph.getNodeWeight(u));
    result *= sigmoid(-norm / weightFactor);
    result /= weightFactor;
    result /= norm;
    return result;
}

TmpCVec<3> SimpleSigmoidNoDimEmbedder::attractionForce(int v, int u) {
    TmpVec<3> result(buffer, 0.0);
    if (v == u) return result;

    CVecRef posV = graph.getPosition(v);
    CVecRef posU = graph.getPosition(u);

    result = posU - posV;
    double norm = result.norm();

    if (norm <= 0) {
        //  displace in random direction if positions are identical
        LOG_WARNING("Random displacement attr V: (" << v << ") U: (" << u << ")");
        result.setToRandomUnitVector();
        return result;
    }

    double weightFactor = idealEdgeLength(graph.getNodeWeight(v), graph.getNodeWeight(u));
    result *= sigmoid(norm / weightFactor);
    result /= weightFactor;
    result /= norm;
    return result;
}

double SimpleSigmoidNoDimEmbedder::sigmoid(double x) { return 1.0 / (1 + std::exp(-x)); }

double SimpleSigmoidNoDimEmbedder::idealEdgeLength(double wu, double wv) { return options.cSpring * wu * wv; }

void SimpleSigmoidWeightedEmbedder::calculateStep() {
    if (insignificantPosChange && !options.staticWeights) {
        insignificantPosChange = false;
        numForceSteps -= 10;
        calculateWeightStep();
    } else {
        calculateForceStep();
    }

    currIteration++;
}

void SimpleSigmoidWeightedEmbedder::calculateWeightStep() {
    numWeightSteps++;
    const int N = graph.getNumVertices();

    double currCooling = std::pow(options.weightCooling, numWeightSteps);
    for (NodeId v = 0; v < N; v++) {
        currentWeightForce[v] = (sumWeightRepulsionForce(v) + sumWeightAttractionForce(v));
        currentWeightForce[v] *= options.weightSpeed * currCooling;
    }

    double sumTmpWeight = 0.0;
    // apply movement based on force
    for (NodeId v = 0; v < N; v++) {
        oldWeights[v] = graph.getNodeWeight(v);
        newWeights[v] = oldWeights[v] + currentWeightForce[v];
        newWeights[v] = std::max(newWeights[v], 10e-100);  // do not allow negative weights

        sumTmpWeight += newWeights[v];
    }

    // normalize weights and update them in graph
    for (NodeId a = 0; a < N; a++) {
        newWeights[a] *= ((double)N / sumTmpWeight);
        // NOTE(JP) scaling the weights seem to perform better
        graph.setNodeWeight(a, newWeights[a]);
    }

    double sumWeightDiff = 0.0;
    double sumNewWeights = 0.0;
    for (NodeId a = 0; a < N; a++) {
        sumWeightDiff += std::abs(newWeights[a] - oldWeights[a]);
        sumNewWeights += newWeights[a];
    }
    double relativeWeightChange = sumWeightDiff / sumNewWeights;

    if (relativeWeightChange < options.relativeWeightMinChange) {
        // LOG_DEBUG( "Insignificant weight change of " << sumWeightDiff / sumNewWeights);
        insignificantWeightChange = true;
    }
}

TmpCVec<2> SimpleSigmoidWeightedEmbedder::repulsionForce(int v, int u) {
    TmpVec<2> result(buffer, 0.0);

    if (v == u) return result;

    CVecRef posV = graph.getPosition(v);
    CVecRef posU = graph.getPosition(u);

    result = posV - posU;
    double norm = result.norm();

    if (norm <= 0) {
        //  displace in random direction if positions are identical
        // LOG_WARNING( "Random displacement rep V: (" << v << ") U: (" << u << ")");
        result.setToRandomUnitVector();
        return result;
    }

    double weightFactor = idealEdgeLength(graph.getNodeWeight(v), graph.getNodeWeight(u));
    result *= sigmoid(options.sigmoidScale * (options.sigmoidLength - norm / weightFactor));
    result /= weightFactor;
    result /= norm;
    result *= options.sigmoidScale;
    return result;
}

TmpCVec<3> SimpleSigmoidWeightedEmbedder::attractionForce(int v, int u) {
    TmpVec<3> result(buffer, 0.0);
    if (v == u) return result;

    CVecRef posV = graph.getPosition(v);
    CVecRef posU = graph.getPosition(u);

    result = posU - posV;
    double norm = result.norm();

    if (norm <= 0) {
        //  displace in random direction if positions are identical
        // LOG_WARNING( "Random displacement attr V: (" << v << ") U: (" << u << ")");
        result.setToRandomUnitVector();
        return result;
    }

    double weightFactor = idealEdgeLength(graph.getNodeWeight(v), graph.getNodeWeight(u));
    result *= sigmoid(options.sigmoidScale * ((norm / weightFactor) - options.sigmoidLength));
    result /= weightFactor;
    result *= options.sigmoidScale;
    result /= norm;
    return result;
}

double SimpleSigmoidWeightedEmbedder::weightRepulsionForce(NodeId v, NodeId u) {
    TmpVec<WEIGHT_REP_BUFFER> tmp(buffer, 0.0);
    if (v == u) return 0.0;

    CVecRef posV = graph.getPosition(v);
    CVecRef posU = graph.getPosition(u);
    double wu = graph.getNodeWeight(u);
    double wv = graph.getNodeWeight(v);

    // assert wu and wv are not nan
    ASSERT(!std::isnan(wu), "Weight of node " << u << " is nan!");
    ASSERT(!std::isnan(wv), "Weight of node " << v << " is nan!");

    tmp = posU - posV;
    double norm = tmp.norm();
    double weightFactor = idealEdgeLength(wu, wv);
    double fract = -norm / weightFactor;
    return sigmoid(fract) * (fract / ((double)options.embeddingDimension) * wu);
}

double SimpleSigmoidWeightedEmbedder::weightAttractionForce(NodeId v, NodeId u) {
    TmpVec<WEIGHT_ATTR_BUFFER> tmp(buffer, 0.0);
    if (v == u) return 0.0;

    CVecRef posV = graph.getPosition(v);
    CVecRef posU = graph.getPosition(u);
    double wu = graph.getNodeWeight(u);
    double wv = graph.getNodeWeight(v);

    // assert wu and wv are not nan
    ASSERT(!std::isnan(wu), "Weight of node " << u << " is nan!");
    ASSERT(!std::isnan(wv), "Weight of node " << v << " is nan!");

    tmp = posU - posV;
    double norm = tmp.norm();
    double weightFactor = idealEdgeLength(wu, wv);
    double fract = norm / weightFactor;
    return sigmoid(fract) * (fract / ((double)options.embeddingDimension) * wu);
}

double SimpleSigmoidWeightedEmbedder::sigmoid(double x) { return 1.0 / (1 + std::exp(-x)); }

double SimpleSigmoidWeightedEmbedder::idealEdgeLength(double wu, double wv) {
    return options.cSpring * std::pow(wu * wv, 1.0 / options.embeddingDimension);
}

void SimpleLinearEmbedder::dumpDebugAtTermination() {
    /*
    std::ofstream file;
    file.open("repForceCalculations.csv");
    file << "Node, degree, Iteration, RepForceCalculations\n";

    for(NodeId v = 0; v < graph.getNumVertices(); v++) {
        for(int iter = 0; iter < currIteration; iter++) {
            file << v << ", " << graph.getNumNeighbors(v) << ", " << iter << ", " <<
    numRepForceCalculations[std::make_pair(v, iter)] << "\n";
        }
    }
    file.close();
    */
    return;
}

TmpCVec<2> SimpleLinearEmbedder::repulsionForce(int v, int u) {
    TmpVec<2> result(buffer, 0.0);
    if (v == u) return result;

    CVecRef posV = graph.getPosition(v);
    CVecRef posU = graph.getPosition(u);

    result = posV - posU;
    double norm = result.norm();

    if (norm <= 0) {
        //  displace in random direction if positions are identical
        LOG_WARNING("Random displacement rep V: (" << v << ") U: (" << u << ")");
        result.setToRandomUnitVector();
        return result;
    }

    double wv = graph.getNodeWeight(v);
    double wu = graph.getNodeWeight(u);
    double similarity = getSimilarity(norm, wu, wv);

    if (similarity > options.sigmoidLength) {
        result *= 0;
    } else {
        result *= options.sigmoidScale / (norm * std::pow(wu * wv, 1.0 / options.embeddingDimension));
        // numRepForceCalculations[std::make_pair(v, currIteration)]++;
    }
    return result;
}

TmpCVec<3> SimpleLinearEmbedder::attractionForce(int v, int u) {
    TmpVec<3> result(buffer, 0.0);
    if (v == u) return result;

    CVecRef posV = graph.getPosition(v);
    CVecRef posU = graph.getPosition(u);

    result = posU - posV;
    double norm = result.norm();

    if (norm <= 0) {
        //  displace in random direction if positions are identical
        LOG_WARNING("Random displacement rep V: (" << v << ") U: (" << u << ")");
        result.setToRandomUnitVector();
        return result;
    }

    double wv = graph.getNodeWeight(v);
    double wu = graph.getNodeWeight(u);
    double similarity = getSimilarity(norm, wu, wv);

    if (similarity <= options.sigmoidLength) {
        result *= 0;
    } else {
        result *= options.sigmoidScale / (norm * std::pow(wu * wv, 1.0 / options.embeddingDimension));
    }
    return result;
}

double SimpleLinearEmbedder::getSimilarity(double norm, double wu, double wv) {
    return norm / std::pow(wu * wv, 1.0 / options.embeddingDimension);
}