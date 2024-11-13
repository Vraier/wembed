#pragma once

#include "Graph.hpp"
#include "Metric.hpp"
#include "EvalOptions.hpp"
#include "Embedding.hpp"
#include "EdgeSampler.hpp"

class EdgeHistogram : public Metric {
   public:
    EdgeHistogram(const OptionValues &options, const Graph &g, Embedding &emb);
    std::vector<std::string> getMetricValues();
    std::vector<std::string> getMetricNames();

   private:
    static void writeHistogram(const OptionValues &options, const std::vector<histEntry> &entries);

    OptionValues options;
    const Graph &graph;
    Embedding &embedding;
};
