#include "GirgGenerator.hpp"

#include "GraphAlgorithms.hpp"
#include "girgs/Generator.h"
#include "EmbeddingIO.hpp"
#include "Macros.hpp"
#include "Rand.hpp"
#include "VecList.hpp"
#include "Toolkit.hpp"

Graph GirgGenerator::generateRandomGraph(OptionValues options) {
    LOG_INFO( "Constructing GIRG...");

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

    // convert the edgeList into a graph
    std::set<std::pair<NodeId, NodeId>> edgeSet;
    for (auto e : edges) {
        std::pair<NodeId, NodeId> reverse = std::make_pair(e.second, e.first);
        edgeSet.insert(e);
        edgeSet.insert(reverse);
    }
    Graph unconnected;
    unconnected.constructFromEdges(edgeSet);
    auto graphAndMap = GraphAlgo::getLargestComponentWithMapping(unconnected);
    Graph connected = graphAndMap.first;
    std::map<NodeId, NodeId> connectedToUnconnected = graphAndMap.second;

    // also output the coordinates (and weights of the girg)
    if (options.girgCoords != "") {
        const int conN = connected.getNumVertices();
        VecList coords(options.genDimension);
        coords.setSize(conN);
        std::vector<double> weights(conN);
        for (NodeId v = 0; v < conN; v++) {
            for (int i = 0; i < options.genDimension; i++) {
                coords[v][i] = girgPositions[connectedToUnconnected[v]][i];
            }
            weights[v] = girgWeights[connectedToUnconnected[v]];
        }

        EmbeddingIO::writeCoordinates(options.girgCoords, coords, weights, Toolkit::createIdentity(conN));
    }

    LOG_INFO( "Finished construction");

    return connected;
}
