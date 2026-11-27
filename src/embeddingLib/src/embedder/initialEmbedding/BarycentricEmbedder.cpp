#include "initialEmbedder/BarycentricEmbedder.hpp"

#include "GraphAlgorithms.hpp"

std::vector<NodeId> BarycentricEmbedder::computeCornerNodes(std::vector<std::vector<double>>& distanceFromCorners) {
    //TODO: Make it general over amount of corners
    std::vector<double>& distancesFromOne = distanceFromCorners[0];
    std::vector<double>& distancesFromTwo = distanceFromCorners[1];
    std::vector<double>& distancesFromThree = distanceFromCorners[2];

    const std::function<double(NodeId, NodeId)> weightF = GraphAlgo::vertexWeightFunc(this->graph, this->opts.embeddingDimension);

    std::vector<double> distances = GraphAlgo::calculateShortestPaths(this->graph, 0, weightF);
    const auto firstCorner = static_cast<NodeId>(
                std::distance(distances.begin(), std::max_element(distances.begin(), distances.end())));
    distances.clear();

    distancesFromOne = GraphAlgo::calculateShortestPaths(this->graph, firstCorner, weightF);
    const auto secondCorner = static_cast<NodeId>(
        std::distance(distancesFromOne.begin(), std::max_element(distancesFromOne.begin(), distancesFromOne.end())));

    distancesFromTwo = GraphAlgo::calculateShortestPaths(this->graph, secondCorner, weightF);

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

    distancesFromThree = GraphAlgo::calculateShortestPaths(this->graph, thirdCorner, weightF);

    return {firstCorner, secondCorner, thirdCorner};
}

void BarycentricEmbedder::computeBarycentricCoords(const std::vector<NodeId>& corners,
                                                   const std::vector<std::vector<double>>& distanceFromCorners) {
    for (int v = 0; v < this->graph.getNumVertices(); v++) {
        if (std::ranges::find(corners, v) != corners.end()) {
            continue;
        }

        std::vector<double> invDist(corners.size());
        double denom = 0.0;
        for (size_t c = 0; c < corners.size(); c++) {
            invDist[c] = 1. / distanceFromCorners[c][v];
            denom += invDist[c];
        }

        std::vector<double> weights(corners.size());
        for (size_t c = 0; c < corners.size(); c++) {
            weights[c] = invDist[c] / denom;
        }

        double xCoord = 0.0;
        double yCoord = 0.0;
        for (size_t w = 0; w < weights.size(); w++) {
            xCoord += weights[w] * initEmb[corners[w]][0];
            yCoord += weights[w] * initEmb[corners[w]][1];
        }

        initEmb[v][0] = xCoord;
        initEmb[v][1] = yCoord;
    }
}

std::vector<std::vector<double> > BarycentricEmbedder::getInitialEmbedding() {

    if (initEmb.empty()) {
        initEmb = std::vector(this->graph.getNumVertices(), std::vector<double>(this->opts.embeddingDimension, 0.0));

        std::vector<std::vector<double>> distanceFromCorners(numberAnchors);

        timer->startTiming("computeCorners", "Compute corner vertices");
        const std::vector<NodeId> corners = computeCornerNodes(distanceFromCorners);
        timer->stopTiming("computeCorners");

        timer->startTiming("EmbedAnchors", "Embed anchor points");
        const double angleGap = 2. / static_cast<double>(numberAnchors);
        // loop over all corner points
        for (size_t i = 0; i < numberAnchors; i++) {
            initEmb[corners[i]][0] = std::cos(static_cast<double>(i) * angleGap * std::numbers::pi_v<double>);
            initEmb[corners[i]][1] = std::sin(static_cast<double>(i) * angleGap * std::numbers::pi_v<double>);
        }
        timer->stopTiming("EmbedAnchors");

        // Embed all other points
        timer->startTiming("computeBary", "Compute barycentric coords");
        computeBarycentricCoords(corners, distanceFromCorners);
        timer->stopTiming("computeBary");
    }

    return initEmb;
}


