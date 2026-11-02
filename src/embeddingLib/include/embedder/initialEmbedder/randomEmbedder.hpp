#pragma once
#include "InitialEmbedderInterface.hpp"

class RandomEmbedder : public InitialEmbedderInterface {

    public:
    explicit RandomEmbedder(const Graph& graph, const EmbedderOptions& opts) : InitialEmbedderInterface(graph, opts) {

    }

    ~RandomEmbedder() override = default;

    std::vector<std::vector<double>> getInitialEmbedding() override;
};
