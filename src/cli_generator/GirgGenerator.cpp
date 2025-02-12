#include "GirgGenerator.hpp"

#include "EmbeddingIO.hpp"
#include "GraphAlgorithms.hpp"
#include "Macros.hpp"
#include "Rand.hpp"
#include "Toolkit.hpp"
#include "girgs/Generator.h"

std::tuple<Graph, std::vector<std::vector<double>>, std::vector<double>> GirgGenerator::generateRandomGraph(
    Options options) {
    LOG_INFO("Constructing GIRG...");

    const int N = options.numNodes;
    const double ple = options.ple;
    const double deg = options.averageDegree;
    const int dim = options.genDimension;
    const double T = options.temperature;
    const bool torus = options.torus;
    const double alpha = T > 0 ? 1 / T : std::numeric_limits<double>::infinity();

    int wSeed = Rand::randomInt(0, 100000);
    int pSeed = Rand::randomInt(0, 100000);
    int sSeed = Rand::randomInt(0, 100000);

    auto girgWeights = girgs::generateWeights(N, ple, wSeed, false);
    auto girgPositions = girgs::generatePositions(N, dim, pSeed);
    girgs::scaleWeights(girgWeights, deg, dim, alpha);

    if (!torus) {
        // scale all positions with 0.5 to prevent wrapping of the torus
        for (auto& pos : girgPositions) {
            for (auto& coordinate : pos) {
                coordinate *= 0.5;
            }
        }
        // scale all weights to accommodate for the lower distances
        double factor = std::pow(0.5, dim);
        for (auto& weight : girgWeights) {
            weight *= factor;
        }
    }

    auto edges = girgs::generateEdges(girgWeights, girgPositions, alpha, sSeed);

    Graph unconnected(edges);
    auto graphAndMap = GraphAlgo::getLargestComponentWithMapping(unconnected);
    Graph connected = graphAndMap.first;
    std::vector<NodeId> connectedToUnconnected = graphAndMap.second;

    // map coordinates and weights to connected graph
    const int conN = connected.getNumVertices();
    std::vector<std::vector<double>> coords(conN, std::vector<double>(dim));
    std::vector<double> weights(conN);
    for (NodeId v = 0; v < conN; v++) {
        for (int i = 0; i < options.genDimension; i++) {
            coords[v][i] = girgPositions[connectedToUnconnected[v]][i];
        }
        weights[v] = girgWeights[connectedToUnconnected[v]];
    }

    LOG_INFO("Finished construction");

    return std::make_tuple(connected, coords, weights);
}
