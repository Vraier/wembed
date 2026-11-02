#pragma once
#include <vector>

#include "EmbedderOptions.hpp"
#include "Graph.hpp"


class InitialEmbedderInterface {
    protected:
    Graph graph;
    EmbedderOptions opts;

    public:
    InitialEmbedderInterface() = delete;
    InitialEmbedderInterface(InitialEmbedderInterface& initEmb) = default;
    InitialEmbedderInterface(InitialEmbedderInterface&&) = default;

    explicit InitialEmbedderInterface(const Graph& graph, const EmbedderOptions& opts) : graph(graph), opts(opts) {}

    InitialEmbedderInterface& operator=(const InitialEmbedderInterface& initEmbed) = default;
    InitialEmbedderInterface& operator=(InitialEmbedderInterface&&) = default;

    virtual ~InitialEmbedderInterface() = default;

    virtual std::vector<std::vector<double>> getInitialEmbedding() = 0;
};
