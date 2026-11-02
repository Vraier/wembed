#pragma once
#include "InitialEmbedderInterface.hpp"

class UniformEmbedder : public InitialEmbedderInterface {

    public:
    explicit UniformEmbedder(const Graph& graph, const EmbedderOptions& opts) : InitialEmbedderInterface(graph, opts) {

    }

    ~UniformEmbedder() override = default;

    std::vector<std::vector<double>> getInitialEmbedding() override;
};