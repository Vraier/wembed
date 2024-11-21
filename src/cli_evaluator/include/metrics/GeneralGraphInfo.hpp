#pragma once

#include "Graph.hpp"
#include "Metric.hpp"
#include "Options.hpp"

/**
 * Calculates general information about the graph.
 * 
 * This includes the number of nodes and edges.
 */
class GeneralGraphInfo : public Metric {
   public:
    GeneralGraphInfo(const Options &options, const Graph &g);

    std::vector<std::string> getMetricValues();
    std::vector<std::string> getMetricNames();


   private:
    Options options;
    const Graph &graph;
};