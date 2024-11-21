#include "GeneralGraphInfo.hpp"

#include "Macros.hpp"

GeneralGraphInfo::GeneralGraphInfo(const Graph& g) : graph(g) {}

std::vector<std::string> GeneralGraphInfo::getMetricValues() {
    std::vector<std::string> result = {std::to_string(graph.getNumVertices()), std::to_string(graph.getNumEdges())};
    return result;
}

std::vector<std::string> GeneralGraphInfo::getMetricNames() {
    std::vector<std::string> result = {"num_nodes", "num_edges"};
    return result;
}
