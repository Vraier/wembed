#pragma once

#include "Graph.hpp"
#include "Metric.hpp"
#include "EvalOptions.hpp"

class GeneralGraphInfo : public Metric {
   public:
    GeneralGraphInfo(const OptionValues &options, const Graph &g);

    std::vector<std::string> getMetricValues();
    std::vector<std::string> getMetricNames();


   private:
    OptionValues options;
    const Graph &graph;
};