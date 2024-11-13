#pragma once

#include "Embedding.hpp"
#include "EvalOptions.hpp"
#include "Graph.hpp"
#include "Metric.hpp"
#include "NodeSampler.hpp"
#include "VecList.hpp"

class Reconstruction : public Metric {
   public:
    Reconstruction(const OptionValues &options, const Graph &g, Embedding &emb);

    std::vector<std::string> getMetricValues();
    std::vector<std::string> getMetricNames();

    const static inline std::vector<int> kVals{1, 2, 5, 10, 100, 200, 500, 1000};

   private:
    static void writeHistogram(const OptionValues &options, const NodeHistogram &entries);

    OptionValues options;
    const Graph &graph;
    Embedding &embedding;

    VecBuffer<1> buffer;
};
