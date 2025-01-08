#include "SimpleSamplingEmbedder.hpp"

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <fstream>
#include <queue>
#include <unordered_set>
#include <omp.h>

#include "Macros.hpp"
#include "WeightedRTree.hpp"

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

int SamplingHeuristic::getNumSamplesForNode(const EmbeddedGraph& graph, NodeId v, int numSamples,
                                            bool uniformSampling) {
    if (uniformSampling) {
        const int numAverageSamples = (double)(graph.getNumEdges() * 2.0 * numSamples) / (double)graph.getNumVertices();
        return std::min(numAverageSamples, graph.getNumVertices() - 1);
    } else {
        return std::min(numSamples * graph.getNumNeighbors(v), graph.getNumVertices() - 1);
    }
}

TmpCVec<AbstractSimpleEmbedder::REP_BUFFER> SimpleSamplingEmbedder::repulsionForce(int v, int u) {
    TmpVec<REP_BUFFER> result(buffer, 0.0);
    if (v == u) return result;

    CVecRef posV = graph.getPosition(v);
    CVecRef posU = graph.getPosition(u);

    // calculate norm according to L_inf or L_2 norm
    double norm;
    result = posV - posU;
    if (options.useInfNorm) {
        norm = result.infNorm();
    } else {
        norm = result.norm();
    }

    if (norm <= 0) {
        result.setToRandomUnitVector();
        return result;
    }

    // calculate the derivative vector
    if (options.useInfNorm) {
        result.infNormed();
    } else {
        result /= norm;
    }

    double wv = graph.getNodeWeight(v);
    double wu = graph.getNodeWeight(u);
    double similarity = getSimilarity(norm, wu, wv);

    if (similarity > options.sigmoidLength) {
        result *= 0;
    } else {
        result *= options.sigmoidScale / (std::pow(wu * wv, 1.0 / options.embeddingDimension));
        numEffectiveRepForceCalculations++;
    }
    return result;
}

TmpCVec<AbstractSimpleEmbedder::ATTR_BUFFER> SimpleSamplingEmbedder::attractionForce(int v, int u) {
    TmpVec<ATTR_BUFFER> result(buffer, 0.0);
    if (v == u) return result;

    CVecRef posV = graph.getPosition(v);
    CVecRef posU = graph.getPosition(u);

    // calculate norm according to L_inf or L_2 norm
    double norm;
    result = posU - posV;
    if (options.useInfNorm) {
        norm = result.infNorm();
    } else {
        norm = result.norm();
    }

    if (norm <= 0) {
        //  displace in random direction if positions are identical
        // LOG_WARNING( "Random displacement rep V: (" << v << ") U: (" << u << ")");
        result.setToRandomUnitVector();
        return result;
    }

    // calculate the derivative vector
    if (options.useInfNorm) {
        result.infNormed();
    } else {
        result /= norm;
    }

    double wv = graph.getNodeWeight(v);
    double wu = graph.getNodeWeight(u);
    double similarity = getSimilarity(norm, wu, wv);

    if (similarity <= options.sigmoidLength) {
        result *= 0;
    } else {
        result *= options.sigmoidScale / (std::pow(wu * wv, 1.0 / options.embeddingDimension));
    }
    return result;
}

double SimpleSamplingEmbedder::getSimilarity(double norm, double wu, double wv) {
    return norm / std::pow(wu * wv, 1.0 / options.embeddingDimension);
}

void SimpleSamplingEmbedder::calculateAllRepellingForces() {

    std::vector<std::vector<NodeId>> repellingCandidates(graph.getNumVertices());
    timer.startTiming("rTree", "Construct RTree");
    RTreeSampling rTree(graph, graph.getAllNodeWeights(), options.sigmoidLength, options.doublingFactor, options.useInfNorm);
    timer.stopTiming("rTree");

    timer.startTiming("candidates", "Find Candidates");
    // i think nodes with a large degree are a big problem here
    // 'dynamic' lets each thread grab a new node as it finished 
    // this helps to balance the load
    #pragma omp parallel for 
    for(NodeId v = 0; v < graph.getNumVertices(); v++) {
        repellingCandidates[v]= rTree.calculateRepellingCandidatesForNode(graph, v, timer);
    }
    timer.stopTiming("candidates");

    timer.startTiming("sum_of_forces", "Compute Sum of Forces for Each Candidate");
    for (NodeId v = 0; v < graph.getNumVertices(); v++) {
        for (NodeId u : repellingCandidates[v]) {
            if (options.neighborRepulsion || !graph.areNeighbors(v, u)) {
                currentForce[v] += repulsionForce(v, u);
            }
        }
    }
    timer.stopTiming("sum_of_forces");



    /* HAAAAAACKKKKK!! 
    // always use the sampling heuristic to calculate the forces
    timer.startTiming("candidates", "Find Candidates");
    numEffectiveRepForceCalculations = 0;
    RepellingCandidates candidates = samplingHeuristic->calculateRepellingCandidates(graph, timer);
    timer.stopTiming("candidates");



    timer.startTiming("sum_of_forces", "Compute Sum of Forces for Each Candidate");
    foundRepForces[currIteration] = candidates.size();
    for (auto [v, u] : candidates) {
        if (options.neighborRepulsion || !graph.areNeighbors(v, u)) {
            currentForce[v] += repulsionForce(v, u);
        }
    }
    correctRepForces[currIteration] = numEffectiveRepForceCalculations;
    timer.stopTiming("sum_of_forces");
    */
}

void SimpleSamplingEmbedder::dumpDebugAtTermination() {
    std::ofstream file;
    file.open("repForcePrecision.csv");
    file << "Iteration, possibleRepForces, foundRepForces, correctRepForces\n";

    for (auto it = possibleRepForces.begin(); it != possibleRepForces.end(); ++it) {
        file << it->first << ", " << possibleRepForces[it->first] << ", " << foundRepForces[it->first] << ", "
             << correctRepForces[it->first] << "\n";
    }
    file.close();

    double repForcePrecision = 0;
    int numValidRepForces = 0;  // can only calculate average when number of rep forces is greater than 0
    for (auto it = possibleRepForces.begin(); it != possibleRepForces.end(); ++it) {
        if (possibleRepForces[it->first] > 0) {
            repForcePrecision += (double)correctRepForces[it->first] / (double)possibleRepForces[it->first];
            numValidRepForces++;
        }
    }
    if (numValidRepForces > 0) {
        repForcePrecision /= (double)numValidRepForces;
    }

    std::cout << "> repForcePrecision=" << repForcePrecision << std::endl;
}

std::unique_ptr<SamplingHeuristic> SimpleSamplingEmbedder::createSamplingHeuristic(SamplingHeuristicType type) {
    SamplingHeuristicType t = static_cast<SamplingHeuristicType>(type);
    const int numNegSample = options.numNegativeSamples;
    const bool uniformSampling = options.uniformSampling;
    const double averageDegree = (double)(graph.getNumEdges() * 2.0) / (double)graph.getNumVertices();

    switch (t) {
        case SamplingHeuristicType::Quadratic:
            return std::make_unique<QuadraticSampling>();
        case SamplingHeuristicType::Random:
            return std::make_unique<RandomSampling>(numNegSample, uniformSampling);
        case SamplingHeuristicType::Girg:
            return std::make_unique<GirgGenSampling>(options.embeddingDimension, numNegSample * averageDegree);
        case SamplingHeuristicType::BFS:
            return std::make_unique<BFSSampling>(graph, numNegSample, uniformSampling);
        case SamplingHeuristicType::Distance:
            return std::make_unique<DistanceSampling>(graph, numNegSample, options.dimensionHint, uniformSampling);
        case SamplingHeuristicType::RTree:
            return std::make_unique<RTreeSampling>(graph, graph.getAllNodeWeights(), options.sigmoidLength,
                                                   options.doublingFactor, options.useInfNorm);
        default:
            LOG_ERROR("Unknown sampling heuristic type ");
            return nullptr;
    }
}

RepellingCandidates QuadraticSampling::calculateRepellingCandidates(const EmbeddedGraph& g, Timer&) {
    RepellingCandidates result;
    for (NodeId v = 0; v < g.getNumVertices(); v++) {
        for (NodeId u = 0; u < g.getNumVertices(); u++) {
            result.push_back(std::make_pair(v, u));
        }
    }
    return result;
}

RepellingCandidates RandomSampling::calculateRepellingCandidates(const EmbeddedGraph& g, Timer&) {
    ASSERT(numNegativeSamples > 0);
    RepellingCandidates result;

    for (NodeId v = 0; v < g.getNumVertices(); v++) {
        // simulates a sparse array with n entries where initially a[i]=i;
        std::unordered_map<int, NodeId> sortedArray;

        int numSamplesForNode = getNumSamplesForNode(g, v, numNegativeSamples, uniformSampling);

        for (int i = 0; i < numSamplesForNode; i++) {
            int randomId = Rand::randomInt(i, g.getNumVertices() - 1);  // random number between i and n-1

            if (sortedArray.find(randomId) == sortedArray.end()) {
                sortedArray[randomId] = randomId;  // make element appear in sortedArray
            }
            if (sortedArray.find(i) == sortedArray.end()) {
                sortedArray[i] = i;  // make element appear in sortedArray
            }
            // swap randomId with i
            std::swap(sortedArray[randomId], sortedArray[i]);
        }

        for (int i = 0; i < numSamplesForNode; i++) {
            result.push_back(std::make_pair(v, sortedArray[i]));
        }
    }
    return result;
}

RepellingCandidates GirgGenSampling::calculateRepellingCandidates(const EmbeddedGraph& g, Timer&) {
    ASSERT(dimension <= 5);

    ASSERT(false, "Not implemented");
    return RepellingCandidates();

    std::vector<double> weights(g.getNumVertices());
    std::vector<std::vector<double>> coords = g.coordinates.convertToVector();
    ASSERT(coords.size() == weights.size());
    ASSERT(coords.size() > 0);
    ASSERT(dimension == coords[0].size());

    // calculate the bounding box of the graph
    std::vector<double> minCoords(dimension, std::numeric_limits<double>::max());
    std::vector<double> maxCoords(dimension, std::numeric_limits<double>::lowest());
    for (int i = 0; i < coords.size(); i++) {
        for (int j = 0; j < dimension; j++) {
            minCoords[j] = std::min(minCoords[j], coords[i][j]);
            maxCoords[j] = std::max(maxCoords[j], coords[i][j]);
        }
    }

    // move the points to all be greater than 0
    for (int i = 0; i < coords.size(); i++) {
        for (int j = 0; j < dimension; j++) {
            coords[i][j] -= minCoords[j];
        }
    }

    // find the largest dimension and scale all coordinates by this amount
    double maxDimExpansion = 0.0;
    for (int j = 0; j < dimension; j++) {
        maxDimExpansion = std::max(maxDimExpansion, maxCoords[j] - minCoords[j]);
    }
    for (int i = 0; i < coords.size(); i++) {
        for (int j = 0; j < dimension; j++) {
            coords[i][j] /= (maxDimExpansion * 2.0);
        }
    }

    // assert correct positions
    for (int i = 0; i < coords.size(); i++) {
        for (int j = 0; j < dimension; j++) {
            ASSERT(coords[i][j] >= 0.0);
            ASSERT(coords[i][j] <= 0.5);
        }
    }

    for (int i = 0; i < weights.size(); i++) {
        weights[i] = g.getNumNeighbors(i);
    }

    // construct the girg
    const double alpha = std::numeric_limits<double>::infinity();
    unused(alpha);

    // girgs::scaleWeights(weights, averageDegree, dimension, alpha);
    double factor = std::pow(0.5, dimension);
    for (auto& weight : weights) {
        weight *= factor;
    }

    // auto edges = girgs::generateEdges(weights, coords, alpha, 1337);
    //  NOTE(JP): i just assume that the generator only outputs one pair for each edge (and not two)
    RepellingCandidates result;
    // for (auto [u, v] : edges) {
    //     result.push_back(std::make_pair(u, v));
    //     result.push_back(std::make_pair(v, u));
    // }

    return result;
}

BFSSampling::BFSSampling(const EmbeddedGraph& g, int numSamples, bool uniformSampling) {
    // do a BFS from every node until enough neighbors are found
    for (NodeId v = 0; v < g.getNumVertices(); v++) {
        int numSamplesForNode = getNumSamplesForNode(g, v, numSamples, uniformSampling);
        std::vector<NodeId> vCandidates;
        std::unordered_set<NodeId> visited;
        std::queue<NodeId> q;
        q.push(v);
        visited.insert(v);
        while (!q.empty() && vCandidates.size() < numSamplesForNode) {
            NodeId u = q.front();
            q.pop();
            for (NodeId w : g.getNeighbors(u)) {
                if (vCandidates.size() >= numSamplesForNode) {
                    break;
                }
                if (visited.find(w) == visited.end()) {
                    visited.insert(w);
                    q.push(w);

                    // only add nodes that are not neighbors
                    if (!g.areNeighbors(v, w) && vCandidates.size() < numSamplesForNode) {
                        vCandidates.push_back(w);
                    }
                }
            }
        }

        // add all the nodes to the repelling candidates
        for (NodeId u : vCandidates) {
            candidates.push_back(std::make_pair(v, u));
        }
    }
}

RepellingCandidates BFSSampling::calculateRepellingCandidates(const EmbeddedGraph& g, Timer&) {
    unused(g);
    return candidates;
}

DistanceSampling::DistanceSampling(const EmbeddedGraph& g, int numSamples, int dimHint, bool uniformSampling) {
    // do a dijkstra from every node until enough neighbors are found
    for (NodeId v = 0; v < g.getNumVertices(); v++) {
        int numSamplesForNode = getNumSamplesForNode(g, v, numSamples, uniformSampling);
        std::vector<NodeId> vCandidates;
        std::unordered_set<NodeId> visited;
        std::priority_queue<std::pair<double, NodeId>, std::vector<std::pair<double, NodeId>>,
                            std::greater<std::pair<double, NodeId>>>
            q;
        q.push(std::make_pair(0.0, v));
        visited.insert(v);
        while (!q.empty() && vCandidates.size() < numSamplesForNode) {
            NodeId u = q.top().second;
            double currDist = q.top().first;
            q.pop();
            for (NodeId w : g.getNeighbors(u)) {
                if (vCandidates.size() >= numSamplesForNode) {
                    break;
                }
                if (visited.find(w) == visited.end()) {
                    double newDist =
                        currDist + std::pow(g.getNumNeighbors(u) * g.getNumNeighbors(w), 1.0 / (double)dimHint);

                    visited.insert(w);
                    q.push(std::make_pair(newDist, w));

                    // only add nodes that are not neighbors
                    if (!g.areNeighbors(v, w) && vCandidates.size() < numSamplesForNode) {
                        vCandidates.push_back(w);
                    }
                }
            }
        }

        // add all the nodes to the repelling candidates
        for (NodeId u : vCandidates) {
            candidates.push_back(std::make_pair(v, u));
        }
    }
}

RepellingCandidates DistanceSampling::calculateRepellingCandidates(const EmbeddedGraph& g, Timer&) {
    unused(g);
    return candidates;
}

RepellingCandidates RTreeSampling::calculateRepellingCandidates(const EmbeddedGraph& g, Timer& timer) {
    using value = std::pair<CVecRef, NodeId>;

    timer.startTiming("construct_rtree", "Construct RTree");
    std::vector<value> values;
    for (NodeId v = 0; v < g.getNumVertices(); v++) {
        values.push_back(std::make_pair(g.getPosition(v), v));
    }

    WeightedRTree rtree(g.getDimension());
    rtree.updateRTree(g.coordinates, g.getAllNodeWeights(), weightBuckets);
    timer.stopTiming("construct_rtree");

    timer.startTiming("query_rtee", "RTree Queries");
    VecBuffer<2> buffer(g.getDimension());


    RepellingCandidates result;
    std::vector<NodeId> vCandidates;
    for (size_t w_class = 0; w_class < rtree.getNumWeightClasses(); w_class++) {
        std::string timing_key("weight_class_");
        timing_key += std::to_string(w_class);
        if (w_class < 10) {
            timer.startTiming(timing_key, std::to_string(w_class) + ". Weight Class");
        } else {
            timer.startTiming("weight_class_remaining", "Remaining Classes");
        }

        for (NodeId v = 0; v < g.getNumVertices(); v++) {
            vCandidates.clear();
            if (useInfNorm) {
                rtree.getNodesWithinWeightedDistanceInfNormForClass(g.getPosition(v), g.getNodeWeight(v), edgeLength,
                                                                    w_class, vCandidates, buffer);
            } else {
                rtree.getNodesWithinWeightedDistanceForClass(g.getPosition(v), g.getNodeWeight(v), edgeLength, w_class,
                                                             vCandidates, buffer);
            }
            for (NodeId u : vCandidates) {
                if (v != u) {
                    result.push_back(std::make_pair(v, u));
                }
            }
        }
        if (w_class < 10) {
            timer.stopTiming(timing_key);
        } else {
            timer.stopTiming("weight_class_remaining");
        }
    }
    timer.stopTiming("query_rtee");
    return result;
}

std::vector<NodeId> RTreeSampling::calculateRepellingCandidatesForNode(const EmbeddedGraph& graph, NodeId v,
                                                                       Timer& timer) const {
    std::vector<NodeId> vCandidates;
    for (size_t w_class = 0; w_class < rtree.getNumWeightClasses(); w_class++) {
        std::vector<NodeId> tmp;
        VecBuffer<2> buffer(graph.getDimension());
        rtree.getNodesWithinWeightedDistanceForClass(graph.getPosition(v), graph.getNodeWeight(v), edgeLength, w_class,
                                                     tmp, buffer);

        for (NodeId u : tmp) {
            if (v != u) {
                vCandidates.push_back(u);
            }
        }
    }
    return vCandidates;
}