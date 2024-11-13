#pragma once

#include "Embedding.hpp"
#include "Graph.hpp"
#include "Metric.hpp"
#include "EvalOptions.hpp"

class EdgeDetection : public Metric {
   public:
    EdgeDetection(const OptionValues &options, const Graph &g, Embedding &emb);

    std::vector<std::string> getMetricValues();
    std::vector<std::string> getMetricNames();

    const static inline std::vector<int> kVals{2, 4, 8, 16, 32, 64, 128, 256, 512};

   private:
    OptionValues options;
    const Graph &graph;
    Embedding &embedding;
};