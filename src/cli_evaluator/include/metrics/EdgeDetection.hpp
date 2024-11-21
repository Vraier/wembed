#pragma once

#include <memory>

#include "Embedding.hpp"
#include "Graph.hpp"
#include "Metric.hpp"
#include "Options.hpp"

/**
 * Calculates the F1 score for predicting edges.
 *
 * Currently only counts the number of nodes and edges.
 */
class EdgeDetection : public Metric {
   public:
    EdgeDetection(const Options &options, const Graph &g, std::shared_ptr<Embedding> embedding);

    std::vector<std::string> getMetricValues();
    std::vector<std::string> getMetricNames();

   private:
    Options options;
    const Graph &graph;
    std::shared_ptr<Embedding> embedding;
};