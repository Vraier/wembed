#include "GirgGenerator.hpp"

#include "EmbeddingIO.hpp"
#include "GraphAlgorithms.hpp"
#include "Macros.hpp"
#include "Rand.hpp"
#include "Toolkit.hpp"
#include "girgs/Generator.h"

std::tuple<Graph, std::vector<std::vector<flt_t>>, std::vector<flt_t>> GirgGenerator::generateRandomGraph(
    Options options) {
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
    auto edges = girgs::generateEdges(girgWeights, girgPositions, alpha, sSeed);
    girgs::scaleWeights(girgWeights, deg, dim, alpha);

    //Transform results to flt_t:
    std::vector<flt_t> girgWeightsPrecTyped(girgWeights.begin(), girgWeights.end());
    std::vector<std::vector<flt_t>> girgPositionsPrecTyped;
    girgPositionsPrecTyped.reserve(girgPositions.size());
    for (const auto& dvec : girgPositions) {
        girgPositionsPrecTyped.emplace_back(dvec.begin(), dvec.end());
    }

    if (!torus) {
        // scale all positions with 0.5 to prevent wrapping of the torus
        for (auto& pos : girgPositionsPrecTyped) {
            for (auto& coordinate : pos) {
                coordinate *= flt_t{0.5};
            }
        }
        // scale all weights to accommodate for the lower distances
        flt_t factor = std::pow(flt_t{0.5}, static_cast<flt_t>(dim));
        for (auto& weight : girgWeightsPrecTyped) {
            weight *= factor;
        }
    }

    Graph unconnected(edges);

    if (!options.unconnected) {
        auto graphAndMap = GraphAlgo::getLargestComponentWithMapping(unconnected);
        Graph connected = graphAndMap.first;
        std::vector<NodeId> connectedToUnconnected = graphAndMap.second;

        // map coordinates and weights to connected graph
        const int conN = connected.getNumVertices();
        std::vector<std::vector<flt_t>> coords(conN, std::vector<flt_t>(dim));
        std::vector<flt_t> weights(conN);
        for (NodeId v = 0; v < conN; v++) {
            for (int i = 0; i < options.genDimension; i++) {
                coords[v][i] = girgPositionsPrecTyped[connectedToUnconnected[v]][i];
            }
            weights[v] = girgWeightsPrecTyped[connectedToUnconnected[v]];
        }

        return std::make_tuple(connected, coords, weights);
    }

    return std::make_tuple(unconnected, girgPositionsPrecTyped, girgWeightsPrecTyped);
}
