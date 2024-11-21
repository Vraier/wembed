#pragma once

#include <string>
#include <vector>

#include "Embedding.hpp"
#include "Graph.hpp"
#include "WeightedGeometric.hpp"


class Metric {
   public:
    virtual ~Metric(){};
    virtual std::vector<std::string> getMetricValues() = 0;
    virtual std::vector<std::string> getMetricNames() = 0;

    static void printCSVToConsole(const std::vector<std::string>& content);
};
