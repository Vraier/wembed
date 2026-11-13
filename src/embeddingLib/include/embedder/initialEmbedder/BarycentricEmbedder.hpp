#pragma once
#include "InitialEmbedderInterface.hpp"

class BarycentricEmbedder : public InitialEmbedderInterface {

    public:
    explicit BarycentricEmbedder(Graph& graph, EmbedderOptions& opts) : InitialEmbedderInterface(graph, opts) {

    }

    ~BarycentricEmbedder() override = default;

    std::vector<std::vector<double>> getInitialEmbedding() override;
};
