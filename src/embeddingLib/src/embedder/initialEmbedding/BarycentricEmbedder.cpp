#include "initialEmbedder/BarycentricEmbedder.hpp"

#include "GraphAlgorithms.hpp"

std::vector<std::vector<double> > BarycentricEmbedder::getInitialEmbedding() {

    std::vector initEmb(this->graph.getNumVertices(), std::vector<double>(this->graph.getNumVertices()));

    //Compute the three corner points
    const std::function<double(NodeId, NodeId)> weightF = GraphAlgo::vertexWeightFunc(this->graph, this->opts.embeddingDimension);

    std::vector<double> distances = GraphAlgo::calculateShortestPaths(this->graph, 0, weightF);
    const auto firstCorner = static_cast<NodeId>(
                std::distance(distances.begin(), std::max_element(distances.begin(), distances.end())));
    distances.clear();

    std::vector<double> distancesFromOne = GraphAlgo::calculateShortestPaths(this->graph, firstCorner, weightF);
    const auto secondCorner = static_cast<NodeId>(
        std::distance(distancesFromOne.begin(), std::max_element(distancesFromOne.begin(), distancesFromOne.end())));

    std::vector<double> distancesFromTwo = GraphAlgo::calculateShortestPaths(this->graph, secondCorner, weightF);

    // Calculate node with the furthest summed distance from first and second corner
    NodeId thirdCorner = 0;
    double maxSummedDist = distancesFromOne[0] + distancesFromTwo[0];
    // TODO: Parallel?
    for (NodeId v = 0; v < this->graph.getNumVertices(); v++) {
        if (v == firstCorner || v == secondCorner) continue;

        if (distancesFromOne[v] + distancesFromTwo[v] > maxSummedDist) {
            thirdCorner = v;
            maxSummedDist = distancesFromOne[v] + distancesFromTwo[v];
        }
    }

    //TODO: Embed three cornerpoints

    //TODO: Embed all other points

    return initEmb;
}


