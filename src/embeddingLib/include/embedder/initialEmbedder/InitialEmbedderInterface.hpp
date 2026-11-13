#pragma once
#include <vector>

#include "EmbedderOptions.hpp"
#include "Graph.hpp"
#include "Timings.hpp"


class InitialEmbedderInterface {
    protected:
    Graph graph;
    EmbedderOptions opts;
    std::shared_ptr<util::Timer> timer;

    public:
    InitialEmbedderInterface() = delete;
    InitialEmbedderInterface(InitialEmbedderInterface& initEmb) = default;
    InitialEmbedderInterface(InitialEmbedderInterface&&) = default;

    explicit InitialEmbedderInterface(const Graph& graph, const EmbedderOptions& opts)
                                      : graph(graph),
                                        opts(opts),
                                        timer(std::make_shared<util::Timer>()) {}

    InitialEmbedderInterface& operator=(const InitialEmbedderInterface& initEmbed) = default;
    InitialEmbedderInterface& operator=(InitialEmbedderInterface&&) = default;

    virtual ~InitialEmbedderInterface() = default;

    virtual std::vector<std::vector<double>> getInitialEmbedding() = 0;

    [[nodiscard]] std::vector<util::TimingResult> getTimings() const {
        std::vector<util::TimingResult> timings = timer->getHierarchicalTimingResults();
        for (auto& result : timings) {
            ++result.depth;
        }
        return timings;
    }
};
