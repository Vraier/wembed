#pragma once

#include <memory>

#include "Embedding.hpp"
#include "Options.hpp"
#include "Graph.hpp"
#include "Metric.hpp"
#include "NodeSampler.hpp"
#include "VecList.hpp"

/**
 * Calculate reconstruction.
 * For every node, calculate the 
 */
class Reconstruction : public Metric {
   public:
    Reconstruction(const Options &options, const Graph &g, std::shared_ptr<Embedding> embedding);

    std::vector<std::string> getMetricValues();
    std::vector<std::string> getMetricNames();

   private:
    // currently unused
    static void writeHistogram(const Options &options, const std::vector<nodeEntry> &entries);

    Options options;
    const Graph &graph;
    std::shared_ptr<Embedding> embedding;

    VecBuffer<1> buffer;
};
