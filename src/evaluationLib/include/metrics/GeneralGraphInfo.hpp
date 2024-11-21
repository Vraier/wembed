#pragma once

#include "Graph.hpp"
#include "Metric.hpp"

/**
 * Calculates general information about the graph.
 * 
 * This includes the number of nodes and edges.
 */
class GeneralGraphInfo : public Metric {
   public:
    GeneralGraphInfo(const Graph &g);

    std::vector<std::string> getMetricValues();
    std::vector<std::string> getMetricNames();


   private:
    const Graph &graph;
};