#pragma once
#include "InitialEmbedderInterface.hpp"

class BarycentricEmbedder : public InitialEmbedderInterface {
    size_t numberAnchors;

    std::vector<NodeId> computeCornerNodes(std::vector<std::vector<double>>& distanceFromCorners);
    void computeBarycentricCoords(const std::vector<NodeId>& corners,
                                  const std::vector<std::vector<double>>& distanceFromCorners);

    public:
    explicit BarycentricEmbedder(const Graph& graph, const EmbedderOptions& opts, size_t corners = 3)
                : InitialEmbedderInterface(graph, opts), numberAnchors(corners) {

    }

    ~BarycentricEmbedder() override = default;

    std::vector<std::vector<double>> getInitialEmbedding() override;
};
