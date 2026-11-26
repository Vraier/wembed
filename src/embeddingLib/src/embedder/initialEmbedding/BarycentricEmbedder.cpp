#include "initialEmbedder/BarycentricEmbedder.hpp"

#include "GraphAlgorithms.hpp"

std::vector<NodeId> BarycentricEmbedder::computeCornerNodes(std::vector<std::vector<double>>& distanceFromCorners) {
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

    const std::vector<double>& distancesFromOne = distanceFromCorners[0];
    const std::vector<double>& distancesFromTwo = distanceFromCorners[1];
    const std::vector<double>& distancesFromThree = distanceFromCorners[2];

    for (int v = 0; v < this->graph.getNumVertices(); v++) {
        if (v == corners[0] || v == corners[1] || v == corners[2]) continue;

        const double invOne = 1. / distancesFromOne[v];
        const double invTwo = 1. / distancesFromTwo[v];
        const double invThree = 1. / distancesFromThree[v];
        const double denom = invOne + invTwo + invThree;

        const double weightOne = invOne / denom;
        const double weightTwo = invTwo / denom;
        const double weightThree = invThree / denom;

        initEmb[v][0] = weightOne * initEmb[corners[0]][0] + weightTwo * initEmb[corners[1]][0] + weightThree * initEmb[corners[2]][0];
        initEmb[v][1] = weightOne * initEmb[corners[0]][1] + weightTwo * initEmb[corners[1]][1] + weightThree * initEmb[corners[2]][1];
    }
}

std::vector<std::vector<double> > BarycentricEmbedder::getInitialEmbedding() {

    if (initEmb.empty()) {
        initEmb = std::vector(this->graph.getNumVertices(), std::vector<double>(this->opts.embeddingDimension, 0.0));

        std::vector<std::vector<double>> distanceFromCorners(3);

        timer->startTiming("computeCorners", "Compute corner vertices");
        const std::vector<NodeId> corners = computeCornerNodes(distanceFromCorners);
        timer->stopTiming("computeCorners");

        // Embed three cornerpoints
        initEmb[corners[0]][0] = 1.0; // first corner at (1,0,...)

        initEmb[corners[1]][0] = std::cos(2./3. * std::numbers::pi_v<double>); // second corner 120° (2/3 pi)
        initEmb[corners[1]][1] = std::sin(2./3. * std::numbers::pi_v<double>); // rotated from first corner

        initEmb[corners[2]][0] = std::cos(4./3. * std::numbers::pi_v<double>); // second corner 240° (4/3 pi)
        initEmb[corners[2]][1] = std::sin(4./3. * std::numbers::pi_v<double>); // rotated from first corner

        // Embed all other points
        timer->startTiming("computeBary", "Compute barycentric coords");
        computeBarycentricCoords(corners, distanceFromCorners);
        timer->stopTiming("computeBary");
    }

    return initEmb;
}


