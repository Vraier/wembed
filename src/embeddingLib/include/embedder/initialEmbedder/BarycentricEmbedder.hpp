#pragma once
#include "InitialEmbedderInterface.hpp"

class BarycentricEmbedder : public InitialEmbedderInterface {

    std::vector<NodeId> computeCornerNodes(std::vector<std::vector<double>>& distanceFromCorners);
    void computeBarycentricCoords(std::vector<std::vector<double>>& initEmb,
                                  const std::vector<NodeId>& corners,
                                  const std::vector<std::vector<double>>& distanceFromCorners) const;

    public:
    explicit BarycentricEmbedder(const Graph& graph, const EmbedderOptions& opts) : InitialEmbedderInterface(graph, opts) {

    }

    ~BarycentricEmbedder() override = default;

    std::vector<std::vector<double>> getInitialEmbedding() override;
};
